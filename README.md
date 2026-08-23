# MiniGPT From Scratch

A small GPT-style language model built from scratch using Python and PyTorch.

The goal of this project is to learn how large language models work by implementing the core components of a Transformer rather than relying on a pre-built language model.

The current model is trained at the character level on *Alice's Adventures in Wonderland*.

## Current Model

The current MiniGPT configuration uses:

- 826,439 trainable parameters
- 4 Transformer blocks
- 4 attention heads per block
- 128-dimensional embeddings
- 128-character context window
- 71-character vocabulary
- Character-level tokenization
- AdamW optimization
- Cross-entropy loss
- CUDA acceleration when available

## Project Structure

```text
llm-project/
│
├── data/
│   └── training.txt
│
├── src/
│   ├── dataset.py
│   ├── generate.py
│   ├── model.py
│   ├── self_attention.py
│   ├── train.py
│   └── transformer_block.py
│
├── demos/
│   ├── attention_demo.py
│   ├── embeddings_demo.py
│   ├── feed_forward.py
│   ├── prepare_data.py
│   └── other learning/demo scripts
│
├── scripts/
│   └── download_dataset.py
│
├── checkpoints/
│
├── .gitignore
└── README.md
```

## How It Works

The basic training pipeline is:

```text
Training Text
     ↓
Character Tokenizer
     ↓
Token IDs
     ↓
Token + Position Embeddings
     ↓
Transformer Blocks
     ↓
Multi-Head Self-Attention
     ↓
Feed-Forward Networks
     ↓
Output Logits
     ↓
Cross-Entropy Loss
     ↓
Backpropagation
     ↓
AdamW
     ↓
Update Model Parameters
```

The model learns by predicting the next character in a sequence.

For example:

```text
Input:   "Alice wa"
Target:  "lice was"
```

Each target is shifted one character forward so that the model learns to predict the next character based only on the characters before it.

## Components

### Character Tokenizer

The tokenizer assigns every unique character in the training dataset an integer ID.

For example:

```text
"Alice"
   ↓
[character token IDs]
```

The current training dataset contains 71 unique characters.

### Embeddings

Token IDs are converted into learned 128-dimensional vectors.

Position embeddings are added so the model can distinguish between characters appearing at different positions in the sequence.

```text
Token Embedding
       +
Position Embedding
       ↓
Transformer Input
```

### Multi-Head Self-Attention

The model uses four attention heads.

The 128-dimensional representation is divided across the four heads:

```text
128 dimensions
      ↓
4 attention heads
      ↓
32 dimensions per head
```

Each head independently calculates Query, Key, and Value representations.

A causal mask prevents the model from seeing future characters while predicting the next character.

### Transformer Blocks

The model currently contains four Transformer blocks.

Each block contains:

```text
Layer Normalization
        ↓
Multi-Head Self-Attention
        ↓
Residual Connection
        ↓
Layer Normalization
        ↓
Feed-Forward Network
        ↓
Residual Connection
```

### Training

Training is performed using:

- Cross-entropy loss
- Backpropagation
- AdamW optimizer
- Random training batches
- Separate training and validation datasets

The current dataset contains approximately:

```text
144,436 total characters

129,992 training characters
14,444 validation characters
```

During the latest experiment, validation loss reached its lowest point at approximately 1,200 training steps before the model began to overfit.

This demonstrated the difference between training performance and generalization to unseen data.

## Text Generation

After training, `generate.py` loads the trained model and generates text autoregressively.

```text
Prompt
   ↓
Model predicts next character
   ↓
Character is sampled
   ↓
Character is appended to prompt
   ↓
Repeat
```

Generation currently supports:

- Temperature sampling
- Top-K sampling
- 128-character context

## Running the Project

Activate the virtual environment:

```cmd
.venv\Scripts\activate
```

Train the model:

```cmd
python src\train.py
```

Generate text:

```cmd
python src\generate.py
```

Test the model architecture:

```cmd
python src\model.py
```

Test the dataset pipeline:

```cmd
python src\dataset.py
```

## Hardware

Development and training are currently performed using an NVIDIA RTX 5070 with CUDA-enabled PyTorch.

PyTorch automatically uses CUDA when an NVIDIA GPU is available.

## What Has Been Implemented

So far, this project has implemented:

- Character-level tokenization
- Dataset encoding and decoding
- Training/validation splitting
- Random batch generation
- Token embeddings
- Positional embeddings
- Query, Key, and Value projections
- Scaled dot-product attention
- Causal masking
- Multi-head self-attention
- Feed-forward neural networks
- Layer normalization
- Residual connections
- Transformer blocks
- GPT-style language model
- Cross-entropy loss
- Backpropagation
- AdamW optimization
- GPU training
- Validation loss tracking
- Temperature sampling
- Top-K sampling
- Autoregressive text generation

## Next Steps

Planned improvements include:

- Model checkpointing
- Saving the best validation model
- Early stopping
- Improved text generation
- Larger datasets
- Subword tokenization
- Larger context windows
- Additional Transformer improvements

## Purpose

This is an educational project intended to build an understanding of how Transformer-based language models work internally.

Rather than treating an LLM as a black box, each major component is implemented and tested individually before being integrated into the full model.
