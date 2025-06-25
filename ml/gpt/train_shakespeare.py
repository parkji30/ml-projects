import torch
import requests
import torch.nn.functional as F
import os
from torch import nn
from model import GPTDecoder
from tqdm import tqdm


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

    batch_size = 65
    seq_length = 128
    iterations = 10000

    model = GPTDecoder(
        vocab_size=len(stoi),
        d_model=768,
        n_heads=8,
        n_layers=40,
        d_ff=2048,
        max_seq_len=seq_length,
    ).to("cuda").to(torch.bfloat16)

    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)

    model_path = "gpt2_model.pth"

    if os.path.exists(model_path):
        state_dict = torch.load(model_path, weights_only=True)
        model.load_state_dict(state_dict['model_state_dict'])
        optim.load_state_dict(state_dict['optim_state_dict'])
        print("\nSucessfully loaded torch model.\n")

    if not args.eval:
        pbar = tqdm(range(iterations), ncols=100, desc="Training")
        for i in pbar:
            context_tensor, response_tensor = create_batch_data(
                train_enc, seq_length, batch_size
            )
            logits, loss = model(context_tensor, response_tensor)

            # Set gradient to 0
            optim.zero_grad()

            # Calculate gradients
            loss.backward()

            # Update parameters
            optim.step()

            # Update progress bar with loss
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

            if i % 100 ==0:
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
