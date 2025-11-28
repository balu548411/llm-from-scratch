# LLM From Scratch

A comprehensive implementation of language models built from scratch using PyTorch. This repository contains implementations of various language models, starting from simple bigram models to full GPT-style transformer architectures.

## Overview

This project demonstrates the step-by-step construction of language models, providing educational implementations that help understand the fundamentals of neural language modeling and transformer architectures.

## Contents

- **biagram.ipynb** - Jupyter notebook for bigram model experimentation
- **gpt.ipynb** - Jupyter notebook for GPT model experimentation
- **gpt-v2.ipynb** - Enhanced GPT model notebook
- **rot_emb-grp_atten.ipynb** - Notebook exploring rotary embeddings and grouped attention
- **input.txt** - Training dataset (Shakespeare text)
- **more.txt** - Generated text output

## Features

### Bigram Model
- Simple character-level language model
- Token embedding table
- Basic text generation capabilities

### GPT Model
- Multi-head self-attention mechanism
- Transformer blocks with layer normalization
- Position embeddings
- Feed-forward networks
- Dropout regularization
- Configurable hyperparameters (embedding dimension, number of heads, number of layers)

## Requirements

- Python 3.x
- PyTorch
- NumPy (if needed)

## Usage

### Using Jupyter Notebooks

Open any of the `.ipynb` files in Jupyter Notebook or JupyterLab to explore and experiment with the models interactively.

## Model Architecture

### Bigram Model
- Simple embedding-based model
- Direct token-to-token prediction
- Minimal context understanding

### GPT Model
- Embedding dimension: 384
- Number of attention heads: 6
- Number of transformer layers: 6
- Block size (context length): 256
- Dropout: 0.2

## Training

Both models are trained on character-level Shakespeare text. The training process includes:
- Train/validation split (90/10)
- Batch processing
- Loss estimation and monitoring
- Text generation after training

## Hyperparameters

### Bigram Model
- Batch size: 32
- Block size: 8
- Learning rate: 1e-2
- Max iterations: 3000

### GPT Model
- Batch size: 64
- Block size: 256
- Learning rate: 3e-4
- Max iterations: 5000
- Evaluation interval: 500

## Generation

After training, both models can generate text by sampling from the learned probability distributions. The GPT model includes context window management to handle sequences longer than the block size.

## Repository

Clone this repository:

```bash
git clone https://github.com/balu548411/llm-from-scratch.git
cd llm-from-scratch
```

Install the required dependencies:
```bash
pip install -r requirements.txt
```

Start the Jupyter Notebook:
```bash
jupyter notebook
```
## License

This project is for educational purposes.

## Notes

- Models use character-level tokenization
- Training data is from the Tiny Shakespeare dataset
- Models are designed for educational understanding rather than production use
- GPU acceleration is supported when available
