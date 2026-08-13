# Project Milestones

## Model 1: Linear regression from scratch

### R1 — Data and forward prediction

Represent advertising-spend and sales data, then compute predictions using supplied model parameters and `y_hat = mx + c`.

### R2 — Prediction error and mean squared error

Calculate individual errors and implement mean squared error without a machine-learning loss function.

### R3 — Analytical gradients

Derive and implement the partial derivatives of MSE with respect to the slope and intercept, then check them numerically.

### R4 — Gradient descent and training loop

Apply one parameter update, expand it into a training loop, and observe the parameters and loss converge.

### R5 — Visualization and evaluation

Plot the observations, fitted line, and loss history; create a train/test split and evaluate test MSE.

### R6 — Noisy data and training behavior

Train on synthetic noisy data and investigate convergence, learning-rate behavior, underfitting, and overfitting.

### R7 — L1 and L2 regularization

Implement both penalties manually and compare their effects on loss, gradients, and learned parameters.

### R8 — SGD and mini-batch training

Implement stochastic and mini-batch gradient descent and compare them with full-batch training.

### R9 — Optimizer progression

Implement Momentum, RMSprop, and Adam incrementally while inspecting their additional state.

### R10 — Library comparison

Compare the manual implementation with scikit-learn and identify which operations the library replaces.

## Model 2: Tiny GPT-style causal language model

### L1 — Dataset and tokenizer

Create the medieval-kingdom text dataset, build a vocabulary, and implement token-to-ID and ID-to-token mappings.

### L2 — Language-model examples

Encode and decode text, create fixed-length sequences, and construct shifted inputs and next-token targets.

### L3 — Tensor foundations and embeddings

Represent batches with PyTorch tensors, inspect shapes, and add token and simple positional embeddings.

### L4 — Single-head causal self-attention

Implement Q, K, V projections, scaled scores, a causal mask, softmax weights, and weighted value aggregation.

### L5 — Multi-head attention

Run multiple attention heads, concatenate their outputs, and apply an output projection.

### L6 — Transformer block

Combine normalization, residual connections, causal attention, and an MLP into one pre-normalized transformer block.

### L7 — Complete language model

Stack transformer blocks, add final normalization and an LM head, and produce vocabulary logits with correct shapes.

### L8 — Loss and backpropagation

Construct causal targets, calculate cross-entropy from logits, run backpropagation, and inspect gradients.

### L9 — Training loop

Train with mini-batches, gradient clipping, Adam, and then AdamW; add weight decay and loss tracking.

### L10 — Autoregressive generation

Generate tokens iteratively, beginning with greedy decoding and later adding temperature, top-k, and top-p separately.

### L11 — Validation and evaluation

Track validation loss, calculate perplexity, generate from fixed prompts, and diagnose overfitting.

### L12 — Training efficiency

Count parameters, estimate training memory, explore numerical precision, and introduce schedules, accumulation, and checkpointing only as needed.

## After the from-scratch models

### A1 — Pretrained-model workflows

Load Hugging Face models and tokenizers, prepare datasets, and understand the components replaced by high-level APIs.

### A2 — Parameter-efficient fine-tuning

Study and apply LoRA, QLoRA, quantization, adapter saving, and adapter merging.

### A3 — Serving and scaling

Study KV caching, batching, latency, throughput, monitoring, distributed training, FSDP, ZeRO, model parallelism, and distillation.

