# ML and LLM From-Scratch Learning Plan

## Goal

Build two models in Python to turn previously studied mathematics and machine-learning theory into practical coding experience:

1. A linear regression model built without a regression library.
2. A tiny GPT-style causal language model built from understandable PyTorch components.

The purpose is to understand each operation, not to finish as quickly as possible.

## Working approach

- Work through one milestone at a time.
- Create files only when the current milestone needs them.
- Study the listed prerequisite topics independently before starting a task.
- Connect each implementation to its mathematical meaning.
- Verify each component with explicit expected behavior and tests.
- Review and correct the learner's implementation before advancing.
- Use conceptual, structural, and pseudocode hints in that order when needed.
- Provide complete implementation code only when explicitly requested.

Mini Python lessons and standalone Python practice exercises are not part of the default workflow.

## Project phases

### Phase 1: Linear regression

Start with data representation and forward prediction, then progressively add loss, gradients, training, evaluation, regularization, and optimizers. Compare with scikit-learn only after the manual implementation is understood.

### Phase 2: Tiny causal language model

Start with text and tokenization, then build embeddings, attention, transformer blocks, training, generation, and evaluation. High-level pretrained-model APIs are excluded during this phase.

### Phase 3: Efficiency and pretrained models

After the tiny model works, study precision, memory, training efficiency, Hugging Face workflows, fine-tuning, LoRA/QLoRA, serving, and distributed training.

## Definition of milestone completion

A milestone is complete when:

- Its required topics have been reviewed independently.
- Every implementation task has been attempted.
- The expected behavior has been demonstrated.
- The listed tests pass.
- The learner can explain the relevant shapes or mathematical operations.
- Important review findings have been corrected.

## Current position

R1–R8 are complete: forward prediction, mean squared error, hand-derived gradients checked numerically, a training loop compared across four learning rates, plots of the fit and the loss curves, a NumPy rewrite trained on seeded noisy data with a train/test split, L1/L2 regularization swept across penalty strengths, and a batching sweep comparing full-batch, mini-batch and SGD.

Next is Regression Milestone R9 in `TASKS.md`: implement Momentum, RMSprop and Adam one at a time, inspecting the extra state each carries.

