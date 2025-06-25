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


def main():
    """Calculate model sizes for the actual configurations used."""
    print("🚀 GPT Model Parameter Calculator")
    print("=" * 50)
    
    # Configuration 1: BPE tokenization (like GPT-2)
    print("\n📊 Configuration 1: BPE Tokenization (GPT-2 style)")
    print("-" * 50)
    config1 = {
        'vocab_size': 50257,  # GPT-2 BPE vocab size
        'd_model': 768,
        'n_layers': 50,
        'n_heads': 12,
        'd_ff': 3072,
        'max_seq_len': 256  # From training config
    }
    
    print(f"Config: {config1}")
    
    # Calculate with original method (incorrect for our architecture)
    params1_orig = calculate_gpt_params(**config1)
    gb1_orig_fp32 = params_to_gb(params1_orig['total_params'], "float32")
    gb1_orig_bf16 = params_to_gb(params1_orig['total_params'], "bfloat16")
    
    # Calculate with corrected SwiGLU method
    params1_correct = calculate_gpt_params_swiglu(**config1)
    gb1_correct_fp32 = params_to_gb(params1_correct['total_params'], "float32")
    gb1_correct_bf16 = params_to_gb(params1_correct['total_params'], "bfloat16")
    
    print(f"\nOriginal calculation (2-layer FFN):")
    print(f"  Total parameters: {params1_orig['total_params']:,}")
    print(f"  Model size (FP32): {gb1_orig_fp32:.2f} GB")
    print(f"  Model size (BF16): {gb1_orig_bf16:.2f} GB")
    
    print(f"\nCorrected calculation (SwiGLU 3-layer FFN):")
    print(f"  Total parameters: {params1_correct['total_params']:,}")
    print(f"  Model size (FP32): {gb1_correct_fp32:.2f} GB")
    print(f"  Model size (BF16): {gb1_correct_bf16:.2f} GB")
    
    # Configuration 2: Character-level tokenization
    print("\n📊 Configuration 2: Character-level Tokenization")
    print("-" * 50)
    config2 = {
        'vocab_size': 100,  # Typical character vocab size (estimate)
        'd_model': 768,
        'n_layers': 24,
        'n_heads': 12,
        'd_ff': 3072,
        'max_seq_len': 256
    }
    
    print(f"Config: {config2}")
    
    # Calculate with corrected SwiGLU method
    params2_correct = calculate_gpt_params_swiglu(**config2)
    gb2_correct_fp32 = params_to_gb(params2_correct['total_params'], "float32")
    gb2_correct_bf16 = params_to_gb(params2_correct['total_params'], "bfloat16")
    
    print(f"\nCorrected calculation (SwiGLU 3-layer FFN):")
    print(f"  Total parameters: {params2_correct['total_params']:,}")
    print(f"  Model size (FP32): {gb2_correct_fp32:.2f} GB")
    print(f"  Model size (BF16): {gb2_correct_bf16:.2f} GB")
    
    # Breakdown for the BPE configuration
    print(f"\n🔍 Parameter Breakdown (BPE Config):")
    print(f"  Embeddings: {params1_correct['embedding_params']:,} ({params1_correct['embedding_params']/params1_correct['total_params']*100:.1f}%)")
    print(f"  Transformer blocks: {params1_correct['transformer_params']:,} ({params1_correct['transformer_params']/params1_correct['total_params']*100:.1f}%)")
    print(f"  Final components: {params1_correct['final_params']:,} ({params1_correct['final_params']/params1_correct['total_params']*100:.1f}%)")
    print(f"  Per block: {params1_correct['per_block_params']:,}")
    
    print(f"\n💡 Key Insights:")
    print(f"  - Using BF16 reduces model size by ~50% compared to FP32")
    print(f"  - The BPE model is larger due to more layers (50 vs 24)")
    print(f"  - SwiGLU adds ~33% more FFN parameters compared to standard FFN")
    print(f"  - Weight tying saves {config1['vocab_size'] * config1['d_model']:,} parameters")


if __name__ == "__main__":
    main()
