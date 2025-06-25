#!/usr/bin/env python3
"""
Download Common Crawl text data for language model training.
Uses the C4 (Colossal Clean Crawled Corpus) dataset from HuggingFace.
"""

import os
import argparse
from datasets import load_dataset
from tqdm import tqdm

def download_c4_subset(output_file="common_crawl_subset.txt", num_examples=100000, split="train"):
    """
    Download a subset of the C4 dataset (cleaned Common Crawl).
    
    Args:
        output_file: Where to save the text data
        num_examples: Number of text examples to download
        split: Dataset split ('train', 'validation')
    """
    print(f"Downloading {num_examples} examples from C4 dataset...")
    print("This may take a while depending on your internet connection.")
    
    # Load the C4 dataset (English)
    # Note: C4 is huge, so we stream it to avoid downloading everything
    dataset = load_dataset("allenai/c4", "en", split=split, streaming=True)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    
    # Download and save text
    with open(output_file, "w", encoding="utf-8") as f:
        pbar = tqdm(total=num_examples, desc="Downloading")
        count = 0
        
        for example in dataset:
            if count >= num_examples:
                break
                
            # C4 examples have a 'text' field
            text = example['text'].strip()
            
            # Skip very short texts
            if len(text) < 100:
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

def download_oscar_subset(output_file="oscar_subset.txt", num_examples=50000):
    """
    Alternative: Download OSCAR dataset (another Common Crawl derivative).
    """
    print(f"Downloading {num_examples} examples from OSCAR dataset...")
    
    # OSCAR is another cleaned Common Crawl corpus
    dataset = load_dataset("oscar", "unshuffled_deduplicated_en", split="train", streaming=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        pbar = tqdm(total=num_examples, desc="Downloading OSCAR")
        count = 0
        
        for example in dataset:
            if count >= num_examples:
                break
                
            text = example['text'].strip()
            
            # Skip very short texts
            if len(text) < 200:
                continue
                
            f.write(text)
            f.write("\n" + "="*50 + "\n")
            
            count += 1
            pbar.update(1)
        
        pbar.close()
    
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
    parser = argparse.ArgumentParser(description="Download Common Crawl text data")
    parser.add_argument("--dataset", choices=["allenai/c4", "oscar"], default="allenai/c4",
                       help="Which dataset to download")
    parser.add_argument("--num_examples", type=int, default=10000,
                       help="Number of text examples to download")
    parser.add_argument("--output", default="common_crawl.txt",
                       help="Output file name")
    parser.add_argument("--convert", action="store_true",
                       help="Convert to tokenizer-friendly format after download")
    
    args = parser.parse_args()
    
    try:
        if args.dataset == "allenai/c4":
            download_c4_subset(args.output, args.num_examples)
        elif args.dataset == "oscar":
            download_oscar_subset(args.output, args.num_examples)
        
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