# Project Tasks

Work on only the current milestone. Later tasks may be refined when earlier implementation decisions are known.

## R1 — Data and forward prediction

### Topics needed

- Python variables and numeric types
- Lists and list length
- `for` loops
- Creating an empty list
- Appending values to a list
- Basic arithmetic and operator precedence
- Printing values
- Optionally, `zip` for traversing related lists

### Tasks

- [x] Create `regression/regression_from_scratch.py`.
- [x] Represent advertising-spend inputs as `1, 2, 3, 4, 5`.
- [x] Represent actual sales as `10, 13, 16, 19, 22`.
- [x] Store the supplied parameters `m = 3` and `c = 7`.
- [x] Create an empty collection for predictions.
- [x] Calculate one prediction per input using `y_hat = mx + c`.
- [x] Store all predictions without hard-coding their values.
- [x] Print each input with its actual and predicted sales.
- [x] Print the complete prediction collection.
- [x] Confirm that the input, target, and prediction collections have equal lengths.

### Expected behavior

The prediction collection should contain `10, 13, 16, 19, 22` in that order.

### Test cases

- `x = 0` produces `7`.
- `x = 2` produces `13`.
- `x = 10` produces `37`.
- Every prediction follows `3x + 7`.
- No NumPy, pandas, scikit-learn, or PyTorch operations are used.

### Review checkpoint

Submit the code and terminal output for review before beginning R2. Be prepared to explain which line implements the multiplication by the slope and which operation adds the intercept.

## R2 — Prediction error and mean squared error

### Topics needed

- Functions, parameters, and return values
- Iterating over paired collections
- Subtraction, exponentiation, summation, and division
- Accumulators
- Input validation and simple exceptions

### Tasks

- [x] Refactor forward prediction into a reusable function.
- [x] Calculate each residual as predicted minus actual.
- [x] Square each residual.
- [x] Implement mean squared error manually.
- [x] Reject or clearly handle collections with different lengths.

### Validation

- Perfect predictions produce MSE `0`.
- Predictions `2, 4` against targets `1, 2` produce MSE `2.5`.
- Changing one target changes the loss in the expected direction.

## R3 — Analytical gradients

### Topics needed

- Partial derivatives and the chain rule
- Functions returning multiple values
- Accumulating values across observations
- Floating-point comparisons
- Numerical finite-difference checks

### Tasks

- [x] Write the MSE gradient equations for `m` and `c` in comments or project notes.
- [x] Implement `dL/dm` and `dL/dc` manually.
- [x] Calculate both gradients for the current parameters.
- [x] Check analytical gradients using small numerical parameter changes.

### Validation

- Analytical and numerical gradients should be close within a chosen tolerance.
- Gradient signs should agree with the direction in which loss changes.

## R4 — Gradient descent and training loop

### Topics needed

- Assignment and parameter updates
- Loops over a fixed number of iterations
- Learning rate
- Recording history
- Basic debugging of diverging numerical values

### Tasks

- [x] Apply one update to `m` and `c`.
- [x] Print parameters and loss before and after the update.
- [x] Build a training loop.
- [x] Record loss and parameter history.
- [x] Experiment with at least three learning rates.

### Validation

- A suitable learning rate lowers loss.
- On clean `3x + 7` data, learned parameters approach `m = 3` and `c = 7`.
- An intentionally excessive learning rate demonstrates instability or divergence.

## R5–R10 — Remaining regression tasks

### Topics needed

- NumPy arrays, shapes, indexing, slicing, and broadcasting
- Random-number generation and reproducibility
- Matplotlib plotting
- Train/test splitting and evaluation
- Regularization derivatives
- Shuffling, batches, and epochs
- Dictionaries or classes for optimizer state
- Package installation and imports

### Tasks

- [x] Convert the working implementation to NumPy after confirming the list-based math.
- [x] Generate reproducible noisy data following approximately `sales = 3x + 7 + noise`.
- [x] Plot data, fitted line, and loss history.
- [x] Add a train/test split and report test MSE.
- [x] Implement and compare L1 and L2 regularization.
- [x] Implement full-batch, stochastic, and mini-batch gradient descent.
- [x] Implement Momentum, RMSprop, and Adam one at a time.
- [x] Compare results with scikit-learn and document what its API replaces.

### Validation

- Tests cover prediction, MSE, gradients, parameter updates, batch boundaries, and regularization.
- Fixed random seeds make comparisons repeatable.
- Plots and metrics support conclusions rather than replacing numerical checks.

## L1–L2 — Text, tokens, and training examples

### Topics needed

- Strings and string methods
- Lists, dictionaries, sets, and tuples
- Sorting and deterministic mappings
- File reading
- Functions and simple classes
- Sequence indexing and slicing

### Tasks

- [x] Create the small medieval-kingdom dataset.
- [x] Build a word-level vocabulary.
- [x] Create token-to-ID and ID-to-token mappings.
- [ ] Implement encode and decode operations.
- [ ] Build fixed-length input sequences and shifted targets.
- [x] Handle unknown tokens or document why the closed dataset does not need them yet.

### Validation

- Decoding an encoded known sentence reproduces its tokens.
- Input and target sequences have equal lengths.
- Each target position is the next token for its corresponding input position.

## L3–L5 — Embeddings and causal attention

### Topics needed

- PyTorch tensors, dtypes, and devices
- Tensor dimensions and shapes
- Indexing, slicing, reshaping, and transposing
- Matrix multiplication and broadcasting
- Embedding lookup
- Softmax and causal masking
- PyTorch modules and parameter initialization

### Tasks

- [ ] Convert token batches to tensors.
- [ ] Build token and positional embeddings.
- [ ] Document shapes at every operation.
- [ ] Implement one causal self-attention head from projections through output.
- [ ] Verify that masked positions receive no attention probability.
- [ ] Extend the implementation to multiple heads and an output projection.

### Validation

- All predicted shapes match the observed tensor shapes.
- Attention rows sum approximately to one.
- A token cannot attend to later tokens.
- Concatenated head dimensions match the model dimension.

## L6–L7 — Transformer block and language model

### Topics needed

- `nn.Module`, `__init__`, and `forward`
- Module composition and module lists
- LayerNorm or RMSNorm
- Residual connections
- Linear layers and activation functions
- Logits and vocabulary projections

### Tasks

- [ ] Add pre-normalization and an attention residual path.
- [ ] Build an MLP and its residual path.
- [ ] Assemble and test one transformer block.
- [ ] Stack a small number of blocks.
- [ ] Add final normalization and the LM head.
- [ ] Return vocabulary logits without applying softmax during training.

### Validation

- The model preserves batch and sequence dimensions.
- Final logits have shape `[batch, sequence, vocabulary]`.
- Residual additions combine tensors of identical shapes.
- Causality still holds through the complete model.

## L8–L9 — Loss and training

### Topics needed

- Cross-entropy and flattened tensor shapes
- PyTorch autograd
- Gradients and optimizer steps
- Zeroing gradients
- Gradient norms and clipping
- Mini-batches, epochs, and shuffling
- Adam and AdamW state
- Weight decay and learning-rate schedules

### Tasks

- [ ] Calculate cross-entropy directly from logits and shifted targets.
- [ ] Run a forward and backward pass.
- [ ] Inspect selected parameter gradients.
- [ ] Add gradient clipping.
- [ ] Build a mini-batch training loop with loss tracking.
- [ ] Train first with Adam and then compare with AdamW.
- [ ] Add warmup or scheduling only after the base loop works.

### Validation

- The initial loss is finite.
- Relevant trainable parameters receive gradients.
- Loss decreases on the tiny training set.
- Gradient clipping enforces the selected norm bound.

## L10–L12 — Generation, evaluation, and efficiency

### Topics needed

- Autoregressive loops
- Probability distributions and sampling
- Temperature, top-k, and top-p
- Train/validation splits
- Perplexity
- Parameter counting and memory arithmetic
- FP32, FP16, and BF16
- Gradient accumulation and checkpointing

### Tasks

- [ ] Implement greedy next-token generation.
- [ ] Generate multiple tokens autoregressively.
- [ ] Add temperature, top-k, and top-p separately.
- [ ] Track training and validation loss.
- [ ] Calculate perplexity and use fixed evaluation prompts.
- [ ] Count parameters and estimate parameter, gradient, optimizer, and activation memory.
- [ ] Explore mixed precision only if the available hardware supports it.
- [ ] Explore gradient accumulation and checkpointing after measuring a need.

### Validation

- Greedy decoding is deterministic for a fixed checkpoint and prompt.
- Sampling respects the configured filtering strategy.
- Validation metrics are computed without parameter updates.
- Memory estimates state their dtype and optimizer assumptions.

## Advanced follow-up tasks

These remain out of scope until all earlier milestones are complete:

- [ ] Load pretrained causal language models and tokenizers with Hugging Face.
- [ ] Prepare a fine-tuning dataset.
- [ ] Compare full fine-tuning with LoRA and QLoRA.
- [ ] Explore quantization and adapter management.
- [ ] Study serving, KV caching, batching, latency, throughput, and monitoring.
- [ ] Study FSDP, DeepSpeed/ZeRO, model parallelism, and distillation.
