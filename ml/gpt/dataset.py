import torch
import numpy as np
from torch.utils.data import Dataset
import tiktoken

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    
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
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
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
        with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
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