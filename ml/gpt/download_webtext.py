#!/usr/bin/env python3
"""
Download FineWeb dataset for language model training.
Uses the FineWeb dataset from HuggingFace (high-quality web text dataset).

FineWeb is a high-quality filtered web text dataset that's often considered 
superior to OpenWebText for training language models.

Usage examples:
    # Download a small subset (10k examples)
    python download_fineweb.py --num_examples 10000 --output fineweb_small.txt
    
    # Download a larger subset (100k examples) 
    python download_fineweb.py --num_examples 100000 --output fineweb_medium.txt
    
    # Download and convert to tokenizer-friendly format
    python download_fineweb.py --num_examples 50000 --convert --output fineweb_ready.txt
"""

import os
import argparse
from datasets import load_dataset
from tqdm import tqdm

# Default data directory (can be customized via command line arguments)
DEFAULT_DATA_DIR = f'/data/'

def download_fineweb_subset(output_file="fineweb_subset.txt", num_examples=100000, split="train"):
    """
    Download a subset of the FineWeb dataset (high-quality web text).
    
    Args:
        output_file: Where to save the text data
        num_examples: Number of text examples to download
        split: Dataset split ('train')
    """
    print(f"Downloading {num_examples} examples from FineWeb dataset...")
    print("This may take a while depending on your internet connection.")
    
    # Load the FineWeb dataset 
    # Note: FineWeb is large, so we stream it to avoid downloading everything
    dataset = load_dataset("HuggingFaceFW/fineweb", split=split, streaming=True)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    
    # Download and save text
    with open(output_file, "w", encoding="utf-8") as f:
        pbar = tqdm(total=num_examples, desc="Downloading FineWeb")
        count = 0
        
        for example in dataset:
            if count >= num_examples:
                break
                
            # FineWeb examples have a 'text' field
            text = example['text'].strip()
            
            # Skip very short texts (FineWeb should have substantial content)
            if len(text) < 200:
                continue
                
            # Write text with separator
            f.write(text)
            f.write("\n" + "="*50 + "\n")  # Separator between documents
            
            count += 1
            pbar.update(1)
        
        pbar.close()
    
    print(f"\nDownloaded {count} text examples to {output_file}")
    
    # Show file size
    file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
    print(f"File size: {file_size:.1f} MB")

def download_fineweb_full(output_file="fineweb_full.txt", max_examples=None):
    """
    Download the full FineWeb dataset (or a large portion of it).
    This provides high-quality web text data for training.
    """
    print("Downloading full FineWeb dataset...")
    print("This will take a significant amount of time and disk space.")
    
    # Load the full FineWeb dataset
    dataset = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        count = 0
        
        for example in dataset:
            if max_examples and count >= max_examples:
                break
                
            text = example['text'].strip()
            
            # Skip very short texts
            if len(text) < 200:
                continue
                
            f.write(text)
            f.write("\n" + "="*50 + "\n")
            
            count += 1
            if count % 1000 == 0:
                print(f"Downloaded {count} examples...")
    
    print(f"\nDownloaded {count} text examples to {output_file}")

def create_tokenizer_friendly_format(input_file, output_file="tokenizer_ready.txt"):
    """
    Convert downloaded text to a format similar to your Shakespeare data.
    Removes document separators and creates one continuous text stream.
    """
    print(f"Converting {input_file} to tokenizer-friendly format...")
    
    with open(input_file, "r", encoding="utf-8") as f_in:
        with open(output_file, "w", encoding="utf-8") as f_out:
            for line in f_in:
                # Skip separator lines
                if line.strip() == "=" * 50:
                    f_out.write("\n\n")  # Double newline between documents
                    continue
                
                # Write the text line
                f_out.write(line)
    
    print(f"Converted text saved to {output_file}")
    
    # Show some stats
    with open(output_file, "r", encoding="utf-8") as f:
        text = f.read()
        print(f"Total characters: {len(text):,}")
        print(f"Unique characters: {len(set(text))}")

def main():
    parser = argparse.ArgumentParser(description="Download FineWeb dataset for language model training")
    parser.add_argument("--mode", choices=["subset", "full"], default="subset",
                       help="Download a subset or attempt full dataset")
    parser.add_argument("--num_examples", type=int, default=10000,
                       help="Number of text examples to download (for subset mode)")
    parser.add_argument("--output", default="fineweb.txt",
                       help="Output file name")
    parser.add_argument("--convert", action="store_true",
                       help="Convert to tokenizer-friendly format after download")
    
    args = parser.parse_args()
    
    try:
        if args.mode == "subset":
            download_fineweb_subset(args.output, args.num_examples)
        elif args.mode == "full":
            download_fineweb_full(args.output)
        
        if args.convert:
            base_name = os.path.splitext(args.output)[0]
            converted_file = f"{base_name}_tokenizer_ready.txt"
            create_tokenizer_friendly_format(args.output, converted_file)
            print(f"\nReady to use with your model: {converted_file}")
            
    except Exception as e:
        print(f"Error downloading data: {e}")
        print("Make sure you have 'datasets' installed: pip install datasets")

if __name__ == "__main__":
    main() 