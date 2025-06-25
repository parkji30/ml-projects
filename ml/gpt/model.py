import torch
import torch.nn as nn
from torch.nn import functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()

        # Linear transformations and split into heads
        Q = (
            self.q_linear(x)
            .view(batch_size, seq_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        K = (
            self.k_linear(x)
            .view(batch_size, seq_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        V = (
            self.v_linear(x)
            .view(batch_size, seq_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply causal mask for autoregressive generation
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, V)

        # Concatenate heads and put through final linear layer
        context = (
            context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        )
        output = self.out(context)

        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_output)

        # Feed-forward with residual connection
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

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

        # Final layer norm and output projection
        self.layer_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)

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
        positions = (
            torch.arange(seq_len, device=context.device)
            .unsqueeze(0)
            .expand(batch_size, seq_len)
        )

        # Embeddings
        token_emb = self.token_embedding(context)
        pos_emb = self.position_embedding(positions)
        x = self.dropout(token_emb + pos_emb)

        # Create causal mask
        mask = self.create_causal_mask(seq_len).to(context.device)

        # Pass through transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)

        # Final layer norm and output projection
        x = self.layer_norm(x)
        logits = self.output_projection(x)

        if response is None:
            return logits
        else:
            # Calculate loss
            B, T, C = logits.shape
            logits = logits.reshape(B * T, C)
            response = response.reshape(B * T)
            loss = F.cross_entropy(logits, response)
            return logits, loss

    def generate(self, context, max_new_tokens, temperature=1.0, top_k=None):
        """Generate new tokens autoregressively"""
        self.eval()

        for _ in range(max_new_tokens):
            # Crop context if it's too long
            context_cond = (
                context
                if context.size(1) <= self.max_seq_len
                else context[:, -self.max_seq_len :]
            )

            # Forward pass
            with torch.no_grad():
                logits = self(context_cond)
                # Get logits for the last token
                logits = logits[:, -1, :] / temperature

                # Apply top-k filtering if specified
                if top_k is not None:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = -float("Inf")

                # Apply softmax to get probabilities
                probs = F.softmax(logits, dim=-1)

                # Sample next token
                next_token = torch.multinomial(probs, num_samples=1)

                # Append to context
                context = torch.cat((context, next_token), dim=1)

        return context
