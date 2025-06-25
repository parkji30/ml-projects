#!/usr/bin/env python3
"""
Calculate parameter count for the Common Crawl GPT model.
"""

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

def main():
    print("=== COMMON CRAWL GPT MODEL PARAMETER CALCULATION ===")
    print()
    
    # Your updated configurations
    seq_length = 256
    
    # Character-level configuration
    print("🔤 CHARACTER-LEVEL TOKENIZATION:")
    print("Configuration:")
    print("  d_model: 768")
    print("  n_layers: 35")
    print("  n_heads: 8") 
    print("  d_ff: 2048")
    print("  seq_length: 256")
    
    # Estimate vocab size for Common Crawl character-level
    # Common Crawl has much more diverse text than Shakespeare
    estimated_char_vocab = 1500  # Reasonable estimate for diverse web text
    
    char_params = calculate_gpt_params(
        vocab_size=estimated_char_vocab,
        d_model=768,
        n_layers=35,
        n_heads=8,
        d_ff=2048,
        max_seq_len=seq_length
    )
    
    print(f"  Estimated vocab_size: {estimated_char_vocab}")
    print()
    print(f"📊 Character-level Results:")
    print(f"  Total parameters: {char_params['total_params']:,}")
    print(f"  Approximately: {char_params['total_params']/1e6:.1f}M parameters")
    print()
    
    # BPE configuration  
    print("🤖 BPE TOKENIZATION:")
    print("Configuration:")
    print("  d_model: 768")
    print("  n_layers: 20") 
    print("  n_heads: 12")
    print("  d_ff: 3072")
    print("  seq_length: 256")
    
    bpe_vocab = 50257  # tiktoken GPT-2 vocabulary
    
    bpe_params = calculate_gpt_params(
        vocab_size=bpe_vocab,
        d_model=768,
        n_layers=20,
        n_heads=12,
        d_ff=3072,
        max_seq_len=seq_length
    )
    
    print(f"  vocab_size: {bpe_vocab}")
    print()
    print(f"📊 BPE Results:")
    print(f"  Total parameters: {bpe_params['total_params']:,}")
    print(f"  Approximately: {bpe_params['total_params']/1e6:.1f}M parameters")
    print()
    
    # Comparisons
    print("📈 COMPARISONS:")
    print(f"  Your Shakespeare model: ~38M parameters")
    print(f"  Your current training: ~221M parameters") 
    print(f"  Character-level Common Crawl: {char_params['total_params']/1e6:.1f}M parameters")
    print(f"  BPE Common Crawl: {bpe_params['total_params']/1e6:.1f}M parameters")
    print()
    print("🏆 VS FAMOUS MODELS:")
    print(f"  GPT-1: 117M parameters")
    print(f"  GPT-2 Small: 124M parameters") 
    print(f"  GPT-2 Medium: 345M parameters")
    print(f"  GPT-2 Large: 774M parameters")
    print()
    
    if char_params['total_params'] > 345e6:
        print("🚀 Your character-level model is LARGER than GPT-2 Medium!")
    elif char_params['total_params'] > 124e6:
        print("🚀 Your character-level model is between GPT-2 Small and Medium!")
    
    if bpe_params['total_params'] > 774e6:
        print("🚀 Your BPE model is LARGER than GPT-2 Large!")
    elif bpe_params['total_params'] > 345e6:
        print("🚀 Your BPE model is between GPT-2 Medium and Large!")

if __name__ == "__main__":
    main() 