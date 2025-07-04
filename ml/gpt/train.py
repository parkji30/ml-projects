#!/usr/bin/env python3
"""
Train GPT model on FineWeb text data with advanced optimizations.
Expected 2-5x speedup compared to baseline training.
Uses FineWeb dataset (high-quality filtered web text).
"""

import torch
import os
from model import GPTDecoder
from tqdm import tqdm
import torch._dynamo
import numpy as np
from torch.utils.data import DataLoader

from dataset import DynamicTextDataset, create_bpe_tokenizer
from plot import create_loss_plotter
from ml.utils import setup_memory_management, print_memory_stats, check_memory_requirements

def main():
    import argparse
    parser = argparse.ArgumentParser(description='GPT training on FineWeb text data with optimizations')
    parser.add_argument('--data_dir', default='.', 
                       help='Directory containing fineweb.txt file')
    parser.add_argument('--eval', action='store_true', 
                       help='Load existing model instead of training')
    parser.add_argument('--use_bpe', action='store_true',
                       help='Use BPE tokenization (only used with --use_text_file)')
    parser.add_argument('--debug', action='store_true',
                       help='Use small batch size and other debug settings')
    parser.add_argument('--safe_mode', action='store_true',
                       help='Use conservative memory settings to prevent OOM')
    args = parser.parse_args()
    
    # Setup optimized memory management
    setup_memory_management()
    print_memory_stats("Initial")
    
    # Memory-optimized configuration
    if args.debug:
        batch_size = 2  # Very small batch for debugging
        seq_length = 128  # Shorter sequences for debugging
        iterations = 50  # Fewer iterations for debugging
        gradient_accumulation_steps = 4  # Smaller accumulation for debugging
        print("🐛 Debug mode enabled - using minimal memory settings")
    elif args.safe_mode:
        batch_size = 8  # Conservative batch size
        seq_length = 512  # Shorter sequences
        iterations = 10000
        gradient_accumulation_steps = 32  # Higher accumulation to maintain effective batch size
        print("🛡️  Safe mode enabled - using conservative memory settings")
    else:
        batch_size = 24  # Reduced from 48
        seq_length = 1024  # Keep long sequences for better context
        iterations = 10000
        gradient_accumulation_steps = 16  # Effective batch size = 384 (reduced from 768)
    
    # Model size configuration - slightly smaller for better memory efficiency
    d_model = 768  # Reduced from 1024
    n_layers = 20  # Reduced from 24
    n_heads = 12   # Reduced from 16
    d_ff = 2048    # Reduced from 3072
    learning_rate = 6e-4

    # Get vocabulary size first
    tokenizer = create_bpe_tokenizer()
    if tokenizer is None:
        raise RuntimeError("tiktoken is required for binary file loading. Install with: pip install tiktoken")
    decode = lambda l: tokenizer.decode(l)
    vocab_size = tokenizer.n_vocab
    print(f"BPE vocab size: {vocab_size}")

    # Check memory requirements and adjust if needed
    batch_size, seq_length = check_memory_requirements(batch_size, seq_length, d_model, n_layers, vocab_size)
    
    print(f"✅ Final config: batch_size={batch_size}, seq_length={seq_length}")
    print(f"   Effective batch size: {batch_size * gradient_accumulation_steps}")
    print(f"   Model: d_model={d_model}, n_layers={n_layers}, vocab_size={vocab_size}")

    # Create datasets with updated settings
    print("🔄 Creating text datasets...")
    
    # Check current directory and data_dir
    print(f"   Current directory: {os.getcwd()}")
    print(f"   Data directory: {args.data_dir}")
    print(f"   Data directory contents: {os.listdir(args.data_dir) if os.path.exists(args.data_dir) else 'Directory does not exist'}")
    
    # Check if fineweb.txt file exists
    text_file = os.path.join(args.data_dir, 'fineweb.txt')
    
    print(f"   Checking for text file...")
    print(f"   - fineweb.txt path: {text_file}")
    print(f"   - fineweb.txt exists: {os.path.exists(text_file)}")
    
    if os.path.exists(text_file):
        text_size = os.path.getsize(text_file)
        print(f"   - fineweb.txt size: {text_size:,} bytes ({text_size/1024/1024/1024:.1f} GB)")
        
        # Quick test to make sure the file is readable
        try:
            with open(text_file, 'r', encoding='utf-8', errors='ignore') as f:
                test_sample = f.read(1000)
                print(f"   - fineweb.txt test: Successfully read {len(test_sample)} characters")
        except Exception as e:
            print(f"   - ❌ fineweb.txt test failed: {e}")
    
    # If file doesn't exist, provide helpful error message
    if not os.path.exists(text_file):
        print("❌ Text file not found!")
        print("💡 Make sure fineweb.txt is in the data directory.")
        print("💡 Or check that you're running from the correct directory.")
        raise FileNotFoundError(f"Text file not found: {text_file}")
    
    try:
        # Create datasets using DynamicTextDataset for fineweb.txt
        print("   - Creating text-based datasets...")
        
        # Create tokenizer encode function
        def tokenizer_encode_fn(text):
            return tokenizer.encode(text) if tokenizer else [0]
        
        # Create train and validation datasets from the same text file
        train_dataset = DynamicTextDataset(
            file_path=text_file,
            sequence_length=seq_length,
            tokenizer_encode_fn=tokenizer_encode_fn,
            split='train',
            train_split=0.9  # 90% for training, 10% for validation
        )
        
        test_dataset = DynamicTextDataset(
            file_path=text_file,
            sequence_length=seq_length,
            tokenizer_encode_fn=tokenizer_encode_fn,
            split='test',
            train_split=0.9  # 90% for training, 10% for validation
        )
        
        print("✅ Text-based datasets created successfully")
        
    except Exception as e:
        print(f"❌ Failed to create datasets: {e}")
        print(f"💡 Make sure fineweb.txt is accessible and readable!")
        print(f"💡 Check that the file is not corrupted and contains valid text!")
        import traceback
        traceback.print_exc()
        raise
    
    # Create DataLoaders with memory-optimized settings
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 0  # Disable multiprocessing to prevent hanging with memory-mapped files

    print(f"🔧 Creating DataLoaders with memory-optimized settings...")
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=False,  # Disable to save memory
        persistent_workers=False,
        prefetch_factor=None,  # Not used when num_workers=0
        drop_last=True  # Ensure consistent batch sizes
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=False,  # Disable to save memory
        persistent_workers=False,
        prefetch_factor=None,  # Not used when num_workers=0
        drop_last=True  # Ensure consistent batch sizes
    )
    
    print(f"✅ DataLoaders created with memory-optimized settings")
    
    # Test dataset access
    print("🔍 Testing dataset access...")
    try:
        print("   - Testing dataset[0]...")
        sample_data = train_dataset[0]
        print(f"✅ Dataset test successful! Sample shape: {sample_data[0].shape}")
        
        print("   - Testing dataset[1]...")
        sample_data2 = train_dataset[1]
        print(f"✅ Dataset[1] test successful! Sample shape: {sample_data2[0].shape}")
        
        print("   - Testing dataset length...")
        dataset_len = len(train_dataset)
        print(f"✅ Dataset length: {dataset_len}")
        
    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Test DataLoader with memory monitoring
    print("🔍 Testing DataLoader with memory monitoring...")
    try:
        print_memory_stats("Before DataLoader test")
        
        # Test DataLoader creation first
        print("   - Testing DataLoader iterator creation...")
        train_iter = iter(train_loader)
        print("   ✅ DataLoader iterator created successfully")
        
        # Test with timeout to prevent hanging
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("DataLoader test timed out - this usually indicates dataset access issues")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(10)  # 10 second timeout (increased from 5)
        
        try:
            print("   - Testing first batch retrieval...")
            test_batch = next(train_iter)
            print(f"✅ DataLoader test successful! Batch shape: {test_batch[0].shape}")
        finally:
            signal.alarm(0)  # Cancel the alarm
            
        print_memory_stats("After DataLoader test")
    except Exception as e:
        print(f"❌ DataLoader test failed: {e}")
        if "timed out" in str(e):
            print("💡 Tip: This timeout usually means there's an issue with the dataset __getitem__ method")
            print("💡 Check if the binary files are corrupted or if there are indexing issues")
        raise

    # Calculate model parameters
    params = calculate_gpt_params(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_seq_len=seq_length,
    )
    print(f"🚀 Total Model Parameters: {params['total_params']:,}")

    # Create model with memory monitoring
    print("🔧 Creating model...")
    print_memory_stats("Before model creation")
    
    model = GPTDecoder(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_seq_len=seq_length,
    ).to(device)
    
    print_memory_stats("After model creation")
    
    # Convert to bfloat16 for memory efficiency
    model = model.to(torch.bfloat16)
    print_memory_stats("After bfloat16 conversion")

    # Conditionally compile model (can increase memory usage)
    if not args.safe_mode:
        print("🔥 Compiling model...")
        print_memory_stats("Before compilation")
        try:
            model = torch.compile(model, mode='reduce-overhead')
            print("✅ Model compiled successfully!")
            print_memory_stats("After compilation")
        except Exception as e:
            print(f"⚠️  Model compilation failed: {e}, continuing without compilation")
    else:
        print("⚠️  Skipping model compilation in safe mode")

    # Initialize optimizer and scheduler
    optim = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=iterations)

    # Initialize loss plotter
    data_type = "fineweb_text" 
    plotter = create_loss_plotter(save_dir=".", model_name=f"GPT2-WebText-{data_type}")
    
    model_path = f"gpt2_webtext_optimized_{data_type}.pth"

    # Load existing model if available
    if os.path.exists(model_path):
        try:
            print("🔄 Loading existing model...")
            checkpoint = torch.load(model_path, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            optim.load_state_dict(checkpoint['optim_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("✅ Successfully loaded existing model")
        except Exception as e:
            print(f"⚠️  Could not load existing model: {e}, starting fresh")

    if not args.eval:
        print(f"🚀 Starting memory-optimized training...")
        print_memory_stats("Before training")
        
        # Create training iterator
        train_iter = iter(train_loader)
        
        # Training loop with enhanced memory management
        pbar = tqdm(range(iterations), ncols=120, desc="Training")
        
        for i in pbar:
            # More frequent memory cleanup
            if i % 20 == 0:
                torch.cuda.empty_cache()
            
            # Gradient accumulation loop
            total_loss = 0
            for micro_step in range(gradient_accumulation_steps):
                try:
                    context_tensor, response_tensor = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    context_tensor, response_tensor = next(train_iter)
                
                # Move to device efficiently
                context_tensor = context_tensor.to(device, non_blocking=True)
                response_tensor = response_tensor.to(device, non_blocking=True)
                
                # Forward pass with memory efficiency
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    logits, loss = model(context_tensor, response_tensor)
                    loss = loss / gradient_accumulation_steps
                
                total_loss += loss.item()
                loss.backward()
                
                # Clear intermediate tensors
                del context_tensor, response_tensor, logits
            
            # Gradient clipping and optimization
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            optim.zero_grad()
            scheduler.step()

            # Update progress bar
            pbar.set_postfix({
                "Loss": f"{total_loss:.4f}", 
                "LR": f"{scheduler.get_last_lr()[0]:.2e}",
                "Mem": f"{torch.cuda.memory_allocated()/1e9:.1f}GB"
            })
            
            # Add to loss plot
            plotter.add_train_loss(i, total_loss)

            # Checkpointing and evaluation with memory cleanup
            if i % 100 == 0:
                torch.cuda.empty_cache()  # Clean up before checkpointing
                
                # Generate sample (with memory management)
                model.eval()
                with torch.no_grad():
                    prompt = "The future of artificial intelligence"
                    
                    if tokenizer is not None:
                        prompt_tokens = tokenizer.encode(prompt)
                    else:
                        prompt_tokens = [0]
                    
                    sample_tokens = model.generate(
                        torch.tensor([prompt_tokens], device=device, dtype=torch.long),
                        max_new_tokens=100,  # Reduced from 150
                        temperature=0.8,
                        top_p=0.9
                    ).tolist()[0]
                    
                    sample_text = decode(sample_tokens)
                
                # Save sample
                with open(f"webtext_samples_optimized_{data_type}.txt", "a", encoding="utf-8") as f:
                    f.write(f"\n--- Iteration {i} (Loss: {total_loss:.4f}) ---\n")
                    f.write(sample_text)
                    f.write("\n" + "="*80 + "\n")
                
                model.train()
                
                # Save model
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optim_state_dict": optim.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "vocab_size": vocab_size,
                    "data_type": data_type,
                    "iteration": i,
                    "loss": total_loss,
                    "config": {
                        "batch_size": batch_size,
                        "seq_length": seq_length,
                        "gradient_accumulation_steps": gradient_accumulation_steps,
                        "d_model": d_model,
                        "n_layers": n_layers,
                        "n_heads": n_heads,
                        "d_ff": d_ff,
                    }
                }, model_path)
                
                print(f"\n💾 Model saved! Sample preview: {sample_text[:50]}...")
                print_memory_stats(f"Checkpoint {i}")
                
            # Evaluation with memory management
            if i % 500 == 0 and i > 0:
                torch.cuda.empty_cache()
                model.eval()
                eval_losses = []
                
                with torch.no_grad():
                    test_iter = iter(test_loader)
                    for eval_step in range(5):  # Reduced from 10
                        try:
                            eval_context, eval_response = next(test_iter)
                            eval_context = eval_context.to(device, non_blocking=True)
                            eval_response = eval_response.to(device, non_blocking=True)
                            
                            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                                _, eval_loss = model(eval_context, eval_response)
                            eval_losses.append(eval_loss.item())
                            
                            # Clean up evaluation tensors
                            del eval_context, eval_response
                            
                        except StopIteration:
                            break
                
                if eval_losses:
                    avg_eval_loss = np.mean(eval_losses)
                    print(f"\n📊 Evaluation at iter {i}: Train Loss: {total_loss:.4f}, Eval Loss: {avg_eval_loss:.4f}")
                    plotter.add_eval_loss(i, avg_eval_loss)
                
                model.train()
                torch.cuda.empty_cache()
                
    else:
        model.eval()
        plotter = create_loss_plotter(save_dir=".", model_name=f"GPT2-WebText-{data_type}")
    
    # Final generation with memory management
    print("\n🎯 === Final Generation ===")
    model.eval()
    torch.cuda.empty_cache()
    
    with torch.no_grad():
        prompt = "The future of artificial intelligence will transform"
        if tokenizer is not None:
            prompt_tokens = tokenizer.encode(prompt)
        else:
            prompt_tokens = [0]
        
        generated = model.generate(
            torch.tensor([prompt_tokens], device=device, dtype=torch.long),
            max_new_tokens=200,  # Reduced from 300
            temperature=0.7,
            top_p=0.9
        ).tolist()[0]
        
        final_text = decode(generated)
        print(final_text)
        
        # Save final generation
        with open(f"final_generation_webtext_{data_type}.txt", "w", encoding="utf-8") as f:
            f.write("=== Final Generation ===\n")
            f.write(final_text)
            f.write("\n")

    # Clean up
    print("📊 Saving final loss plot...")
    plotter.save_final_plot(f"final_gpt2_webtext_{data_type}_loss_curve.png")
    plotter.close()
    
    print("\n🎉 Training completed successfully!")
    print_memory_stats("Final")


if __name__ == "__main__":
    main() 