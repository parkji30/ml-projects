#!/usr/bin/env python3
"""
Train GPT model on Common Crawl data instead of Shakespeare.
"""

import torch
import torch.nn.functional as F
import os
from torch import nn
from model import GPTDecoder
from tqdm import tqdm

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

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

def create_batch_data(data, sequence_length, batch_size):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    idxes = torch.randint(len(data) - sequence_length, (batch_size, 1))
    context_tensor = torch.stack(
        [torch.tensor(data[i : i + sequence_length], device=device, dtype=torch.long) for i in idxes]
    )
    response_tensor = torch.stack(
        [
            torch.tensor(data[i + 1 : i + sequence_length + 1], device=device, dtype=torch.long)
            for i in idxes
        ]
    )
    return context_tensor, response_tensor

def main():
    import argparse
    parser = argparse.ArgumentParser(description='GPT training on Common Crawl')
    parser.add_argument('--data', default='common_crawl_tokenizer_ready.txt', 
                       help='Path to Common Crawl text file')
    parser.add_argument('--eval', action='store_true', 
                       help='Load existing model instead of training')
    parser.add_argument('--use_bpe', action='store_true',
                       help='Use BPE tokenization instead of character-level')
    args = parser.parse_args()
    
    print(f"Loading data from {args.data}...")
    
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

    # Model configuration - adjusted for your hardware
    batch_size = 32  # Keep smaller for your T4
    seq_length = 256  # Can use longer sequences with Common Crawl
    iterations = 10000

    # Model size - you can experiment with these
    if args.use_bpe:
        # Larger model for BPE (since vocab is much bigger)
        d_model = 768
        n_layers = 20  # Start smaller for BPE
        n_heads = 12
        d_ff = 3072
    else:
        # Your current char-level config
        d_model = 768
        n_layers = 35  # Can use more layers with char-level
        n_heads = 8
        d_ff = 2048

    print(f"Model config: d_model={d_model}, n_layers={n_layers}, vocab_size={vocab_size}")

    model = GPTDecoder(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_seq_len=seq_length,
    ).to('cuda').to(torch.bfloat16)
    

    optim = torch.optim.AdamW(model.parameters(), lr=6e-4)  # Lower LR for bigger model

    model_path = "gpt2_common_crawl.pth"

    if os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, weights_only=True)
            model.load_state_dict(state_dict['model_state_dict'])
            optim.load_state_dict(state_dict['optim_state_dict'])
            print("\nSuccessfully loaded existing model.\n")
        except:
            print("Could not load existing model, starting fresh.")

    print("Compiling Model...")
    model = torch.compile(model)
    print('Done')

    if not args.eval:
        print("Starting training...")
        pbar = tqdm(range(iterations), ncols=100, desc="Training")
        for i in pbar:
            context_tensor, response_tensor = create_batch_data(
                train_enc, seq_length, batch_size
            )
            logits, loss = model(context_tensor, response_tensor)

            optim.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optim.step()

            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

            if i % 100 == 0:
                # Generate sample text
                model.eval()
                with torch.no_grad():
                    if args.use_bpe:
                        # For BPE, start with a simple prompt
                        prompt = "The"
                        prompt_tokens = encode(prompt)
                    else:
                        # For char-level, start with random character
                        prompt_tokens = [torch.randint(vocab_size, (1,)).item()]
                    
                    sample_tokens = model.generate(
                        torch.tensor([prompt_tokens], device='cuda', dtype=torch.long),
                        max_new_tokens=100,
                    ).tolist()[0]
                    
                    sample_text = decode(sample_tokens)
                    
                with open("common_crawl_samples.txt", "a", encoding="utf-8") as f:
                    f.write(f"\n--- Iteration {i} ---\n")
                    f.write(sample_text)
                    f.write("\n" + "="*50 + "\n")
                
                model.train()
                
                # Save model
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optim_state_dict": optim.state_dict(),
                    "vocab_size": vocab_size,
                    "use_bpe": args.use_bpe,
                }, model_path)
                print(f"Model saved. Sample: {sample_text[:100]}...")
    else:
        model.eval()
    
    # Generate some final text
    print("\n=== Final Generation ===")
    model.eval()
    with torch.no_grad():
        if args.use_bpe:
            prompt = "The future of artificial intelligence"
            prompt_tokens = encode(prompt)
        else:
            prompt_tokens = [torch.randint(vocab_size, (1,)).item()]
        
        generated = model.generate(
            torch.tensor([prompt_tokens], device='cuda', dtype=torch.long),
            max_new_tokens=200,
        ).tolist()[0]
        
        final_text = decode(generated)
        print(final_text)

if __name__ == "__main__":
    main() 