import torch
import torch.nn as nn
from torch.nn import functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, use_flash_attention=True):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.use_flash_attention = use_flash_attention

        # Fused QKV projection for efficiency
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.dropout_p = dropout

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()

        # Fused QKV computation
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention: (batch, seq_len, n_heads, d_k)
        q = q.view(batch_size, seq_len, self.n_heads, self.d_k)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_k)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_k)

        if self.use_flash_attention:
            # Use PyTorch's built-in Flash Attention (PyTorch 2.0+)
            # scaled_dot_product_attention automatically uses Flash Attention when available
            context = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,  # We use causal mask built into the function
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=True  # This enables causal masking automatically
            )
            # Output shape: (batch, seq_len, n_heads, d_k)
            # Use reshape instead of view to handle non-contiguous tensors
            context = context.reshape(batch_size, seq_len, d_model)
        else:
            # Standard scaled dot-product attention with manual optimizations
            # Reshape to (batch, n_heads, seq_len, d_k) for manual attention
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            context = self._standard_attention(q, k, v, mask)

        output = self.out(context)
        return output

    def _standard_attention(self, q, k, v, mask=None):
        batch_size, n_heads, seq_len, d_k = q.size()
        
        # More memory-efficient attention computation
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        # Apply causal mask for autoregressive generation
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        else:
            # Create causal mask if not provided
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool))
            scores = scores.masked_fill(~causal_mask, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, v)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return context


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        # Use SwiGLU activation for better performance
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False) 
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # SwiGLU: swish(xW1) ⊙ (xW3) then linear projection
        return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-norm architecture for better training stability
        attn_output = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_output)

        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_output)

        return x


class GPTDecoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=256,
        n_heads=8,
        n_layers=6,
        d_ff=1024,
        max_seq_len=1024,
        dropout=0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Embeddings with weight tying
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        self.emebedding_drop = nn.Dropout(p=dropout)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

        # Final layer norm
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Weight tying: share weights between input embedding and output projection
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        self.output_projection.weight = self.token_embedding.weight

        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def create_causal_mask(self, seq_len):
        """Create a causal mask to prevent attention to future tokens"""
        mask = torch.tril(torch.ones(seq_len, seq_len))
        return mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions

    def forward(self, context, response=None):
        batch_size, seq_len = context.size()

        # Create position indices
        positions = torch.arange(seq_len, device=context.device).unsqueeze(0).expand(batch_size, seq_len)

        # Embeddings
        token_emb = self.token_embedding(context)
        pos_emb = self.position_embedding(positions)
        x = self.dropout(token_emb + pos_emb)
        x = self.emebedding_drop(x)

        # Pass through transformer blocks (mask handled internally now)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        # Final layer norm and output projection
        x = self.layer_norm(x)
        logits = self.output_projection(x)

        if response is None:
            return logits
        else:
            # Calculate loss with label smoothing for better generalization
            B, T, C = logits.shape
            logits_flat = logits.reshape(B * T, C)
            response_flat = response.reshape(B * T)
            
            # Label smoothing
            smoothing = 0.1
            confidence = 1.0 - smoothing
            log_probs = F.log_softmax(logits_flat, dim=-1)
            nll_loss = F.nll_loss(log_probs, response_flat, reduction='mean')
            smooth_loss = -log_probs.mean(dim=-1).mean()
            loss = confidence * nll_loss + smoothing * smooth_loss
            
            return logits, loss

    @torch.inference_mode()
    def generate(self, context, max_new_tokens, temperature=1.0, top_k=None, top_p=0.9):
        """Generate new tokens autoregressively with top-p sampling"""
        self.eval()

        for _ in range(max_new_tokens):
            # Crop context if it's too long
            context_cond = (
                context
                if context.size(1) <= self.max_seq_len
                else context[:, -self.max_seq_len :]
            )

            # Forward pass
            logits = self(context_cond)
            # Get logits for the last token
            logits = logits[:, -1, :] / temperature

            # Apply top-k filtering if specified
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("Inf")

            # Apply top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float("Inf")

            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to context
            context = torch.cat((context, next_token), dim=1)

        return context


def calculate_gpt_params(vocab_size, d_model, n_layers, n_heads, d_ff, max_seq_len):
    """Calculate total parameters for GPT model."""
    
    # Embeddings
    token_embedding = vocab_size * d_model
    position_embedding = max_seq_len * d_model
    embedding_params = token_embedding + position_embedding
    
    # Per transformer block parameters
    # Attention: Q, K, V, Out linear layers (each d_model x d_model + bias)
    attention_params = 4 * (d_model * d_model + d_model)
    # Feed forward: two linear layers
    ff_params = (d_model * d_ff + d_ff) + (d_ff * d_model + d_model)
    # Layer norms: 2 per block, each with weight + bias
    layernorm_params = 2 * (d_model + d_model)
    
    per_block_params = attention_params + ff_params + layernorm_params
    total_transformer_params = per_block_params * n_layers
    
    # Final components
    final_layernorm = d_model + d_model
    output_projection = d_model * vocab_size + vocab_size
    final_params = final_layernorm + output_projection
    
    total_params = embedding_params + total_transformer_params + final_params
    
    return {
        'embedding_params': embedding_params,
        'transformer_params': total_transformer_params,
        'final_params': final_params,
        'total_params': total_params,
        'per_block_params': per_block_params
    }


def calculate_gpt_params_swiglu(vocab_size, d_model, n_layers, n_heads, d_ff, max_seq_len):
    """Calculate total parameters for GPT model with SwiGLU architecture (used in actual model)."""
    
    # Embeddings
    token_embedding = vocab_size * d_model
    position_embedding = max_seq_len * d_model
    embedding_params = token_embedding + position_embedding
    
    # Per transformer block parameters
    # Attention: QKV fused projection (3 * d_model * d_model, no bias) + output projection (d_model * d_model, no bias)
    attention_params = 3 * d_model * d_model + d_model * d_model  # QKV + out projection, no bias
    
    # Feed forward with SwiGLU: 3 linear layers (w1, w2, w3), all without bias
    # w1: d_model -> d_ff, w2: d_ff -> d_model, w3: d_model -> d_ff
    ff_params = (d_model * d_ff) + (d_ff * d_model) + (d_model * d_ff)
    
    # Layer norms: 2 per block, each with weight + bias (d_model parameters each)
    layernorm_params = 2 * (d_model + d_model)  # weight + bias for each norm
    
    per_block_params = attention_params + ff_params + layernorm_params
    total_transformer_params = per_block_params * n_layers
    
    # Final components
    final_layernorm = d_model + d_model  # weight + bias
    # Output projection shares weights with token embedding (weight tying), so no additional params
    output_projection = 0  # Weight tying means no additional parameters
    final_params = final_layernorm + output_projection
    
    total_params = embedding_params + total_transformer_params + final_params
    
    return {
        'embedding_params': embedding_params,
        'transformer_params': total_transformer_params,
        'final_params': final_params,
        'total_params': total_params,
        'per_block_params': per_block_params
    }


def params_to_gb(total_params, dtype="float32"):
    """Convert parameter count to GB based on data type."""
    bytes_per_param = {
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
        "int8": 1
    }
    
    bytes_total = total_params * bytes_per_param.get(dtype, 4)
    gb = bytes_total / (1024**3)
    return gb