import torch
from model import GPTDecoder

def count_parameters():
    # Updated model configuration
    model = GPTDecoder(
        vocab_size=65, 
        d_model=768, 
        n_heads=8, 
        n_layers=40, 
        d_ff=2048, 
        max_seq_len=128
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"=== UPDATED MODEL PARAMETER COUNT ===")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Approximately: {total_params/1e6:.1f}M parameters")
    print(f"")
    print(f"Previous model: ~38M parameters")
    print(f"This model is {total_params/38e6:.1f}x larger!")
    print(f"")
    
    # Configuration summary
    print("=== MODEL CONFIGURATION ===")
    print(f"d_model: 768 (was 512)")
    print(f"n_layers: 40 (was 18)")
    print(f"d_ff: 2048 (was 1024)")
    print(f"vocab_size: 65 (same)")
    print(f"max_seq_len: 128 (was 256)")

    # Let's also break down by component
    print("\nParameter breakdown:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.numel():,}")

if __name__ == "__main__":
    count_parameters() 