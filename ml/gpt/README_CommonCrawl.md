# Training GPT on Common Crawl Data

This guide shows you how to download and train your GPT model on Common Crawl data instead of Shakespeare.

## Quick Start

### 1. Install Dependencies
First, make sure you have the required packages:
```bash
pip install datasets tiktoken
```

### 2. Download Common Crawl Data
```bash
# Download 10,000 examples (~50MB)
python download_common_crawl.py --num_examples 10000 --convert

# Download more data (100,000 examples ~500MB)
python download_common_crawl.py --num_examples 100000 --convert

# Use OSCAR dataset instead
python download_common_crawl.py --dataset oscar --num_examples 50000 --convert
```

### 3. Train Your Model

**Character-level (like your Shakespeare model):**
```bash
python train_on_common_crawl.py --data common_crawl_tokenizer_ready.txt
```

**BPE tokenization (like real GPT models):**
```bash
python train_on_common_crawl.py --data common_crawl_tokenizer_ready.txt --use_bpe
```

## What's Different from Shakespeare Training?

### Dataset Size
- **Shakespeare**: ~1MB, 65 unique characters
- **Common Crawl**: 50MB-500MB+, 1000+ unique characters or 50K BPE tokens

### Vocabulary 
- **Character-level**: ~1000-2000 unique characters (vs 65 for Shakespeare)
- **BPE**: 50,257 tokens (same as GPT-2)

### Model Configuration
The scripts automatically adjust model size based on tokenization:

**Character-level:**
- vocab_size: ~1000-2000
- d_model: 768
- n_layers: 20
- Parameters: ~200M

**BPE:**
- vocab_size: 50,257  
- d_model: 768
- n_layers: 12 (smaller due to larger vocab)
- Parameters: ~400M+

## Memory Requirements

Your Tesla T4 (16GB) can handle:
- **Character-level**: batch_size=32, seq_length=256
- **BPE**: batch_size=16, seq_length=128 (due to larger vocab)

## Expected Results

**Character-level training** will generate text that looks more like natural English but with some artifacts:
```
The company announced today that it will be expanding its operations...
```

**BPE training** will generate higher quality text but requires more memory:
```
The future of artificial intelligence depends on our ability to create systems that...
```

## Dataset Options

### C4 (Colossal Clean Crawled Corpus)
- Used by T5, PaLM, and other major models  
- Pre-cleaned and filtered
- English only
- **Recommended for most use cases**

### OSCAR
- Larger, multilingual
- Less filtered
- Use if you want more diverse/raw text

## Performance Expectations

Training for 5000 iterations should give you a reasonable model that can generate coherent English text instead of just Shakespeare-style language.

## Troubleshooting

**Out of Memory?**
- Reduce batch_size to 16 or 8
- Reduce seq_length to 128
- Use character-level instead of BPE

**Slow downloads?**
- Start with fewer examples (1000-5000)
- Check your internet connection
- The datasets library caches data locally

**Poor text quality?**
- Train for more iterations (10K+)
- Use BPE tokenization
- Increase model size if you have memory 