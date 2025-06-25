#!/usr/bin/env python3
"""
Train GPT model on Common Crawl data with advanced optimizations.
Expected 2-5x speedup compared to baseline training.
"""

import torch
import os
from model import GPTDecoder
from tqdm import tqdm
import torch._dynamo
import numpy as np

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

from calculate_params import calculate_gpt_params


class OptimizedDataLoader:
    """Optimized data loader for faster batch generation"""
    def __init__(self, data, sequence_length, batch_size, device):
        self.data = torch.tensor(data, dtype=torch.long, device=device)
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.device = device
        self.data_len = len(data)
        
        # Pre-compute valid starting indices
        self.valid_indices = torch.arange(self.data_len - sequence_length, device=device)
        
    def get_batch(self):
        # Sample random starting indices
        idx = torch.randint(len(self.valid_indices), (self.batch_size,), device=self.device)
        start_indices = self.valid_indices[idx]
        
        # Vectorized batch creation
        indices = start_indices.unsqueeze(1) + torch.arange(self.sequence_length, device=self.device)
        target_indices = start_indices.unsqueeze(1) + torch.arange(1, self.sequence_length + 1, device=self.device)
        
        context = self.data[indices]
        targets = self.data[target_indices]
        
        return context, targets


def create_bpe_tokenizer():
    """Use OpenAI's BPE tokenizer for better handling of diverse text."""
    if not TIKTOKEN_AVAILABLE:
        print("tiktoken not available, falling back to character-level")
        return None
    
    try:
        # Use GPT-2 tokenizer (50,257 tokens)
        tokenizer = tiktoken.get_encoding("gpt2")
        return tokenizer
    except Exception as e:
        print(f"Error loading tiktoken: {e}, falling back to character-level")
        return None
    

def create_char_data_mappers(input_file_path):
    """Character-level tokenization (like your Shakespeare model)."""
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"Data file {input_file_path} not found. Run download_common_crawl.py first!")

    with open(input_file_path, "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(list(set(text)))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}

    return stoi, itos, text


def create_train_test_loader(text, split=0.9):
    train_length = int(split * len(text))
    train_dataset = text[:train_length]
    test_dataset = text[train_length:]
    return train_dataset, test_dataset


def main():
    import argparse
    parser = argparse.ArgumentParser(description='GPT training on Common Crawl with optimizations')
    parser.add_argument('--data', default='common_crawl_tokenizer_ready.txt', 
                       help='Path to Common Crawl text file')
    parser.add_argument('--eval', action='store_true', 
                       help='Load existing model instead of training')
    parser.add_argument('--use_bpe', action='store_true',
                       help='Use BPE tokenization instead of character-level')
    args = parser.parse_args()
    
    print(f"🚀 Loading data from {args.data}...")
    
    if args.use_bpe:
        # BPE tokenization (like real GPT models)
        tokenizer = create_bpe_tokenizer()
        if tokenizer is None:
            print("BPE tokenizer not available, using character-level")
            args.use_bpe = False
    
    if not args.use_bpe:
        # Character-level tokenization (like your Shakespeare model)
        stoi, itos, text_data = create_char_data_mappers(args.data)
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: "".join([itos[i] for i in l])
        vocab_size = len(stoi)
        print(f"Character-level vocab size: {vocab_size}")
    else:
        # BPE tokenization
        with open(args.data, "r", encoding="utf-8") as f:
            text_data = f.read()
        
        if tokenizer is not None:
            encode = lambda s: tokenizer.encode(s)
            decode = lambda l: tokenizer.decode(l)
            vocab_size = tokenizer.n_vocab
            print(f"BPE vocab size: {vocab_size}")
        else:
            raise RuntimeError("BPE tokenizer failed to load")

    train, test = create_train_test_loader(text_data)
    train_enc = encode(train)
    test_enc = encode(test)
    
    print(f"Training data size: {len(train_enc):,} tokens")

    # A100 optimized configuration with bfloat16
    batch_size = 128  # Larger batch size for A100 with bfloat16
    seq_length = 256  # Longer sequences for better context
    iterations = 10000
    gradient_accumulation_steps = 4  # Effective batch size = 512
    
    # Initialize optimized data loaders
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader = OptimizedDataLoader(train_enc, seq_length, batch_size, device)
    test_loader = OptimizedDataLoader(test_enc, seq_length, batch_size, device)

    # Model size configuration based on tokenization
    if args.use_bpe:
        # Optimized config for BPE (larger vocab)
        d_model = 768
        n_layers = 50  # Balanced for speed vs quality
        n_heads = 12
        d_ff = 3072
        learning_rate = 3e-4  # Lower LR for larger vocab
    else:
        # Optimized config for character-level
        d_model = 768
        n_layers = 24  # Balanced configuration
        n_heads = 12
        d_ff = 3072
        learning_rate = 6e-4

    print(f"✅ Model config: d_model={d_model}, n_layers={n_layers}, vocab_size={vocab_size}")

    params = calculate_gpt_params(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_seq_len=seq_length,
    )
    print(f"🚀 Total Number of Model Params is {params['total_params']}.")

    model = GPTDecoder(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_seq_len=seq_length,
    ).to(device).to(torch.bfloat16)

    # Compile model for faster execution (PyTorch 2.0+)
    print("🔥 Compiling model with bfloat16 for A100...")
    model = torch.compile(model)
    print("✅ Model compiled successfully with bfloat16!")

    
    # Optimizer with weight decay
    optim = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=iterations)

    model_path = "gpt2_common_crawl_optimized.pth"

    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            optim.load_state_dict(checkpoint['optim_state_dict'])
            # Mixed precision disabled
            # if 'scaler_state_dict' in checkpoint:
            #     scaler.load_state_dict(checkpoint['scaler_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("✅ Successfully loaded existing model with optimizations.\n")
        except Exception as e:
            print(f"⚠️  Could not load existing model: {e}, starting fresh.")

    if not args.eval:
        print("🚀 Starting optimized training...")
        pbar = tqdm(range(iterations), ncols=120, desc="Training")
        
        for i in pbar:
            # Gradient accumulation loop
            total_loss = 0
            for micro_step in range(gradient_accumulation_steps):
                context_tensor, response_tensor = train_loader.get_batch()
                
                # bfloat16 forward pass (optimized for A100)
                logits, loss = model(context_tensor, response_tensor)
                loss = loss / gradient_accumulation_steps  # Scale loss
                
                total_loss += loss.item()
                
                # Standard backward pass
                loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step
            optim.step()
            optim.zero_grad()
            scheduler.step()

            # Update progress bar with detailed metrics
            try:
                gpu_util = f"{torch.cuda.utilization()}%" if torch.cuda.is_available() else "N/A"
            except:
                gpu_util = "N/A"  # pynvml not available
                
            pbar.set_postfix({
                "Loss": f"{total_loss:.4f}", 
                "LR": f"{scheduler.get_last_lr()[0]:.2e}",
                "GPU": gpu_util
            })

            if i % 100 == 0:
                # Generate sample text
                model.eval()
                with torch.no_grad():
                    if args.use_bpe:
                        # For BPE, start with a meaningful prompt
                        prompt = "The future of artificial intelligence"
                        prompt_tokens = encode(prompt)
                    else:
                        # For char-level, start with a common character
                        prompt_tokens = [encode("T")[0]] if encode("T") else [0]
                    
                    sample_tokens = model.generate(
                        torch.tensor([prompt_tokens], device=device, dtype=torch.long),
                        max_new_tokens=150,
                        temperature=0.8,
                        top_p=0.9
                    ).tolist()[0]
                    
                    sample_text = decode(sample_tokens)
                    
                with open("common_crawl_samples_optimized.txt", "a", encoding="utf-8") as f:
                    f.write(f"\n--- Iteration {i} (Loss: {total_loss:.4f}) ---\n")
                    f.write(sample_text)
                    f.write("\n" + "="*80 + "\n")
                
                model.train()
                
                # Save model with optimization states (mixed precision disabled)
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optim_state_dict": optim.state_dict(),
                    # "scaler_state_dict": scaler.state_dict(),  # Mixed precision disabled
                    "scheduler_state_dict": scheduler.state_dict(),
                    "vocab_size": vocab_size,
                    "use_bpe": args.use_bpe,
                    "iteration": i,
                    "loss": total_loss,
                }, model_path)
                
                print(f"\n💾 Model saved! Sample: {sample_text[:100]}...")
                
            # Additional evaluation every 500 iterations
            if i % 500 == 0 and i > 0:
                model.eval()
                eval_losses = []
                
                # Evaluate on test set
                with torch.no_grad():
                    for eval_step in range(10):  # Quick evaluation
                        eval_context, eval_response = test_loader.get_batch()
                        # bfloat16 evaluation (optimized for A100)
                        _, eval_loss = model(eval_context, eval_response)
                        eval_losses.append(eval_loss.item())
                
                avg_eval_loss = np.mean(eval_losses)
                print(f"\n📊 Evaluation at iter {i}: Train Loss: {total_loss:.4f}, Eval Loss: {avg_eval_loss:.4f}")
                
                model.train()
                
    else:
        model.eval()
    
    # Generate final high-quality text
    print("\n🎯 === Final Generation ===")
    model.eval()
    with torch.no_grad():
        if args.use_bpe:
            prompt = "The future of artificial intelligence will transform"
            prompt_tokens = encode(prompt)
        else:
            prompt = "The"
            prompt_tokens = encode(prompt)
        
        generated = model.generate(
            torch.tensor([prompt_tokens], device=device, dtype=torch.long),
            max_new_tokens=300,
            temperature=0.7,
            top_p=0.9
        ).tolist()[0]
        
        final_text = decode(generated)
        print(final_text)
        
        # Save final generation
        with open("final_generation_common_crawl.txt", "w", encoding="utf-8") as f:
            f.write("=== Final Generation ===\n")
            f.write(final_text)
            f.write("\n")

    print("\n🎉 Training completed successfully!")

if __name__ == "__main__":
    main() 