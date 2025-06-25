import torch
import requests
import torch.nn.functional as F
import os
from torch import nn
from model import GPTDecoder
from tqdm import tqdm
# from torch.amp import autocast, GradScaler  # Disabled mixed precision
import torch._dynamo
import numpy as np


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


def create_data_mappers(input_file_path):
    input_file_path = os.path.join(input_file_path)

    if not os.path.exists(input_file_path):
        data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        with open(input_file_path, "w") as f:
            f.write(requests.get(data_url).text)

    with open(input_file_path, "r") as f:
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='GPT training and eval')
    parser.add_argument('--eval', action='store_true', help='Load existing model instead of training')
    args = parser.parse_args()
    
    ## Let's Group our code here Together
    stoi, itos, text_data = create_data_mappers("input.txt")
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])

    train, test = create_train_test_loader(text_data)
    train_enc = encode(train)
    test_enc = encode(test)

    batch_size = 256  # A100 optimized with bfloat16
    seq_length = 512  # Longer sequences for better context
    iterations = 10000
    gradient_accumulation_steps = 2  # Effective batch size = 256 * 2 = 512
    
    # Initialize optimized data loaders
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader = OptimizedDataLoader(train_enc, seq_length, batch_size, device)
    test_loader = OptimizedDataLoader(test_enc, seq_length, batch_size, device)

    model = GPTDecoder(
        vocab_size=len(stoi),
        d_model=768,
        n_heads=8,
        n_layers=40,
        d_ff=2048,
        max_seq_len=seq_length,
    ).to("cuda").to(torch.bfloat16)  # Native bfloat16 for A100

    # Compile model for faster execution (PyTorch 2.0+)
    model = torch.compile(model)

    # Mixed precision disabled
    # scaler = GradScaler()
    
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=iterations)

    model_path = "gpt2_model.pth"

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optim_state_dict'])
                    # Mixed precision disabled
            # if 'scaler_state_dict' in checkpoint:
            #     scaler.load_state_dict(checkpoint['scaler_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("\nSucessfully loaded torch model.\n")

    if not args.eval:
        pbar = tqdm(range(iterations), ncols=100, desc="Training")
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
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step
            optim.step()
            optim.zero_grad()
            scheduler.step()

            # Update progress bar with loss
            pbar.set_postfix({
                "Loss": f"{total_loss:.4f}", 
                "LR": f"{scheduler.get_last_lr()[0]:.2e}"
            })

            if i % 100 == 0:
                with open("evals.txt", "a") as f:
                    f.write('\n')
                    f.write(f"\nEval at Iteration {i}\n")
                    f.write('-' * 50 + '\n')

                    output = decode(
                        model.generate(
                            torch.tensor([torch.randint(len(stoi), (1,))], device='cuda', dtype=torch.long).unsqueeze(0),
                            max_new_tokens=100,
                        ).tolist()[0]
                    )
                    f.write(output)

                    print("\n")
                
                # Save the trained model
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optim_state_dict": optim.state_dict(),
                    # "scaler_state_dict": scaler.state_dict(),  # Mixed precision disabled
                    "scheduler_state_dict": scheduler.state_dict(),
                }, model_path)
                print("Model saved as 'gpt2_model.pth'")
    else:
        model.eval()
    
    # Some testing
    print(
        decode(
            model.generate(
                torch.tensor([torch.randint(len(stoi), (1,))], device='cuda', dtype=torch.long).unsqueeze(0),
                max_new_tokens=100,
            ).tolist()[0]
        )
    )
