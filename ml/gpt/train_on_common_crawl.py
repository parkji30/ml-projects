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
from torch.utils.data import Dataset, DataLoader
import mmap

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

from calculate_params import calculate_gpt_params


class DynamicTextDataset(Dataset):
    """
    Dynamic Dataset that reads data from disk without loading everything into memory.
    Uses memory mapping for efficient file access.
    """
    def __init__(self, file_path, sequence_length, tokenizer_encode_fn, split='train', train_split=0.9):
        self.file_path = file_path
        self.sequence_length = sequence_length
        self.encode = tokenizer_encode_fn
        self.split = split
        
        # Get file size and calculate split boundaries
        with open(file_path, 'r', encoding='utf-8') as f:
            # Read a small sample to estimate token-to-char ratio
            sample = f.read(10000)  # Read 10k chars for estimation
            f.seek(0, 2)  # Go to end of file
            self.file_size = f.tell()
        
        # Estimate tokens per character (rough approximation)
        sample_tokens = len(self.encode(sample))
        self.chars_per_token = len(sample) / sample_tokens if sample_tokens > 0 else 1
        
        # Calculate split boundaries in bytes
        train_end_byte = int(self.file_size * train_split)
        
        if split == 'train':
            self.start_byte = 0
            self.end_byte = train_end_byte
        else:  # test
            self.start_byte = train_end_byte
            self.end_byte = self.file_size
        
        # Estimate number of sequences we can extract
        section_size = self.end_byte - self.start_byte
        # Conservative estimate: assume we need sequence_length * chars_per_token characters per sequence
        chars_needed_per_seq = int(sequence_length * self.chars_per_token * 1.2)  # 20% buffer
        self.estimated_sequences = max(1, section_size // chars_needed_per_seq)
        
        print(f"Dataset {split}: {section_size:,} bytes, estimated {self.estimated_sequences:,} sequences")
    
    def __len__(self):
        return self.estimated_sequences
    
    def __getitem__(self, idx):
        """
        Dynamically read a sequence from the file.
        Returns input_ids and target_ids tensors.
        """
        with open(self.file_path, 'r', encoding='utf-8') as f:
            # Calculate a random starting position within our split
            section_size = self.end_byte - self.start_byte
            # Reserve space for reading sequence + 1 token
            max_chars_needed = int((self.sequence_length + 1) * self.chars_per_token * 2)  # Conservative estimate
            
            # Ensure we don't read past our section boundary
            max_start = max(self.start_byte, self.end_byte - max_chars_needed)
            if max_start <= self.start_byte:
                start_pos = self.start_byte
            else:
                # Use idx to create some determinism while still having variety
                np.random.seed(idx)
                start_pos = np.random.randint(self.start_byte, max_start)
            
            f.seek(start_pos)
            
            # Read a chunk of text
            chunk = f.read(max_chars_needed)
            
            if len(chunk) == 0:
                # Fallback: read from the beginning of our section
                f.seek(self.start_byte)
                chunk = f.read(max_chars_needed)
            
            # Tokenize the chunk
            tokens = self.encode(chunk)
            
            # If we don't have enough tokens, pad or repeat
            if len(tokens) < self.sequence_length + 1:
                # Repeat the tokens to get enough length
                repeats_needed = (self.sequence_length + 1) // len(tokens) + 1
                tokens = (tokens * repeats_needed)[:self.sequence_length + 1]
            
            # Extract input and target sequences
            input_ids = torch.tensor(tokens[:self.sequence_length], dtype=torch.long)
            target_ids = torch.tensor(tokens[1:self.sequence_length + 1], dtype=torch.long)
            
            return input_ids, target_ids


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
    

def create_char_tokenizer(input_file_path):
    """Character-level tokenization (like your Shakespeare model)."""
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"Data file {input_file_path} not found. Run download_common_crawl.py first!")

    # Read a sample to build vocabulary
    print("Building character vocabulary from file sample...")
    chars = set()
    with open(input_file_path, "r", encoding="utf-8") as f:
        # Read file in chunks to build vocabulary without loading everything
        chunk_size = 1024 * 1024  # 1MB chunks
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            chars.update(set(chunk))
            
            # Stop after reading enough to get a good vocabulary sample
            if len(chars) > 1000 or f.tell() > 10 * 1024 * 1024:  # Stop after 10MB or 1000 unique chars
                break

    chars = sorted(list(chars))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}

    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])

    return encode, decode, len(stoi)


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
    
    print(f"🚀 Setting up dynamic data loading from {args.data}...")
    
    if args.use_bpe:
        # BPE tokenization (like real GPT models)
        tokenizer = create_bpe_tokenizer()
        if tokenizer is None:
            print("BPE tokenizer not available, using character-level")
            args.use_bpe = False
    
    if not args.use_bpe:
        # Character-level tokenization (like your Shakespeare model)
        encode, decode, vocab_size = create_char_tokenizer(args.data)
        print(f"Character-level vocab size: {vocab_size}")
    else:
        # BPE tokenization
        if tokenizer is not None:
            encode = lambda s: tokenizer.encode(s)
            decode = lambda l: tokenizer.decode(l)
            vocab_size = tokenizer.n_vocab
            print(f"BPE vocab size: {vocab_size}")
        else:
            raise RuntimeError("BPE tokenizer failed to load")

    # A100 optimized configuration with bfloat16
    batch_size = 128  # Larger batch size for A100 with bfloat16
    seq_length = 256  # Longer sequences for better context
    iterations = 10000
    gradient_accumulation_steps = 4  # Effective batch size = 512
    
    # Create dynamic datasets
    print("🔄 Creating dynamic datasets...")
    train_dataset = DynamicTextDataset(args.data, seq_length, encode, split='train')
    test_dataset = DynamicTextDataset(args.data, seq_length, encode, split='test')
    
    # Create DataLoaders with multiple workers for faster loading
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 4 if device == "cuda" else 2  # Adjust based on your system
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True if device == "cuda" else False,
        persistent_workers=True if num_workers > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True if device == "cuda" else False,
        persistent_workers=True if num_workers > 0 else False
    )
    
    print(f"✅ Dynamic DataLoaders created with {num_workers} workers")

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
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("✅ Successfully loaded existing model with optimizations.\n")
        except Exception as e:
            print(f"⚠️  Could not load existing model: {e}, starting fresh.")

    if not args.eval:
        print("🚀 Starting optimized training with dynamic data loading...")
        
        # Create iterator for the DataLoader
        train_iter = iter(train_loader)
        
        pbar = tqdm(range(iterations), ncols=120, desc="Training")
        
        for i in pbar:
            # Gradient accumulation loop
            total_loss = 0
            for micro_step in range(gradient_accumulation_steps):
                try:
                    # Get next batch from DataLoader
                    context_tensor, response_tensor = next(train_iter)
                except StopIteration:
                    # Restart iterator when we reach the end
                    train_iter = iter(train_loader)
                    context_tensor, response_tensor = next(train_iter)
                
                # Move to device
                context_tensor = context_tensor.to(device, non_blocking=True)
                response_tensor = response_tensor.to(device, non_blocking=True)
                
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
                
                # Save model with optimization states
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optim_state_dict": optim.state_dict(),
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
                    test_iter = iter(test_loader)
                    for eval_step in range(10):  # Quick evaluation
                        try:
                            eval_context, eval_response = next(test_iter)
                            eval_context = eval_context.to(device, non_blocking=True)
                            eval_response = eval_response.to(device, non_blocking=True)
                            
                            # bfloat16 evaluation (optimized for A100)
                            _, eval_loss = model(eval_context, eval_response)
                            eval_losses.append(eval_loss.item())
                        except StopIteration:
                            break
                
                if eval_losses:
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