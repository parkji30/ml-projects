import torch
import os

def block_print(func):
    """Decorator that wraps function output with dashed lines"""
    def wrapper(*args, **kwargs):
        print('-' * 50)
        result = func(*args, **kwargs)
        print('-' * 50)
        return result
    return wrapper

@block_print
def setup_memory_management():
    """Setup optimized memory management for A100"""
    if torch.cuda.is_available():
        # Set expandable segments with more aggressive settings
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:256'
        
        # Enable memory-efficient attention if available
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            print("✅ Flash Attention enabled for memory efficiency")
        except:
            print("⚠️  Flash Attention not available")
        
        try:
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            print("✅ Memory-efficient attention enabled")
        except:
            print("⚠️  Memory-efficient attention not available")
        
        # Clear cache and reset peak memory stats
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Set memory fraction to avoid OOM
        torch.cuda.set_per_process_memory_fraction(0.95)
        
        print(f"🔧 Memory management setup complete")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"   Memory fraction set to 95%")

@block_print
def check_memory_requirements(batch_size, seq_length, d_model, n_layers, vocab_size):
    """Estimate memory requirements and suggest adjustments"""
    if not torch.cuda.is_available():
        return batch_size, seq_length
    
    # Rough memory estimation (in GB)
    # Model parameters
    model_params = vocab_size * d_model + n_layers * (4 * d_model * d_model + 2 * d_model)
    model_memory = model_params * 2 / 1e9  # bfloat16 = 2 bytes per param
    
    # Activations memory (per batch)
    activation_memory = batch_size * seq_length * d_model * n_layers * 4 / 1e9  # rough estimate
    
    # Total memory needed
    total_memory = model_memory + activation_memory * 2  # x2 for gradients
    
    # Available memory
    available_memory = torch.cuda.get_device_properties(0).total_memory / 1e9 * 0.9  # 90% of GPU
    
    print(f"📊 Memory estimation:")
    print(f"   Model: {model_memory:.1f} GB")
    print(f"   Activations: {activation_memory:.1f} GB")
    print(f"   Total needed: {total_memory:.1f} GB")
    print(f"   Available: {available_memory:.1f} GB")
    
    if total_memory > available_memory:
        print("⚠️  Estimated memory usage exceeds available memory!")
        
        # Suggest reductions
        new_batch_size = max(1, int(batch_size * available_memory / total_memory))
        new_seq_length = seq_length
        
        if new_batch_size < 4:  # If batch size gets too small, reduce sequence length
            new_seq_length = max(256, int(seq_length * 0.75))
            new_batch_size = max(4, int(batch_size * available_memory / total_memory * 1.5))
        
        print(f"💡 Suggested adjustments:")
        print(f"   Reduce batch_size from {batch_size} to {new_batch_size}")
        if new_seq_length != seq_length:
            print(f"   Reduce seq_length from {seq_length} to {new_seq_length}")
        
        return new_batch_size, new_seq_length
    
    return batch_size, seq_length

@block_print
def print_memory_stats(stage):
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"📊 {stage} - GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        

