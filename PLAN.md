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

**Scalars before matrices.** Each component is built twice: first explicitly, with plain Python numbers and loops on a hand-sized example covering the whole end-to-end path, and only then in tensor form. The explicit version is the reference the tensor version is verified against — the same sequence Phase 1 used, where lists and loops carried R1–R4 and NumPy arrived at R6 only after the arithmetic was proven. The matrix form is the destination, not the starting point.

### Phase 3: Efficiency and pretrained models

After the tiny model works, study precision, memory, training efficiency, Hugging Face workflows, fine-tuning, LoRA/QLoRA, serving, and distributed training.

## Tooling: deliberately deferred

Platform and MLOps tooling is learned against a problem, not in advance. Decisions
made so far, so they are not re-litigated:

- **Hugging Face** — first used as a *consumer* at A1, loading pretrained models
  and tokenizers. Nothing built before then is worth publishing: the regression
  artifact is two floats, and the tiny LLM is a teaching artifact. The first
  thing worth putting on the Hub is a **LoRA adapter from A2**.
- **MLflow** — introduced when experiment tracking earns its place, which is
  around **L9** (many runs, minutes each, several hyperparameters) or A1–A2. Until
  then the `docs/qa-notes-*.md` files record the numbers adequately, and adopting the tool
  before feeling the problem it solves does not stick.
- **Databricks** — out of scope unless a specific role requires it. It is a
  commercial platform, not a concept; nothing here approaches the scale that
  motivates it. The underlying idea, distributed processing, is met at A3 through
  FSDP/ZeRO from the training side.

## Definition of milestone completion

A milestone is complete when:

- Its required topics have been reviewed independently.
- Every implementation task has been attempted.
- The expected behavior has been demonstrated.
- The listed tests pass.
- The learner can explain the relevant shapes or mathematical operations.
- Important review findings have been corrected.

## Current position

R1–R10 are complete, closing Phase 1: forward prediction, mean squared error, hand-derived gradients checked numerically, a training loop compared across four learning rates, plots of the fit and the loss curves, a NumPy rewrite trained on seeded noisy data with a train/test split, L1/L2 regularization swept across penalty strengths, a batching sweep comparing full-batch, mini-batch and SGD, Momentum/RMSprop/Adam implemented by hand, and a scikit-learn comparison that agrees to fourteen digits.

L1 is complete, opening Phase 2: the medieval-kingdom corpus, a word-level vocabulary of 14 tokens, and the token-to-ID / ID-to-token mappings, with the 34.1% majority-class baseline recorded before any model exists.

Next is Milestone L2 in `TASKS.md`: implement encode and decode, then build fixed-length input sequences with shifted next-token targets. Still no PyTorch — L2 is lists and integer IDs.

