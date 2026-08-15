# RegLLM — ML and LLM models built from scratch

A hands-on learning project. Two models, both written by hand, to turn studied
mathematics and machine-learning theory into coding experience:

1. **Linear regression** — no regression library. Data representation, forward
   prediction, MSE, analytical gradients, gradient descent, regularization, and
   optimizers, all implemented directly.
2. **A tiny GPT-style causal language model** — tokenizer, embeddings, causal
   self-attention, multi-head attention, transformer blocks, training,
   generation, and evaluation, built from understandable PyTorch parts.

The point is to understand every operation, not to finish quickly. No
pretrained models, no high-level training APIs, and no library call standing in
for a concept that has not yet been implemented by hand.

## Documents

| File | Purpose |
| --- | --- |
| [PLAN.md](PLAN.md) | Goal, working approach, phases, definition of milestone completion |
| [MILESTONES.md](MILESTONES.md) | The milestone list — R1–R10, L1–L12, A1–A3 |
| [TASKS.md](TASKS.md) | Current milestone detail: topics, tasks, validation, checkboxes |
| [CLAUDE.md](CLAUDE.md) | Rules for any AI agent working in this repo |
| [Codex.md](Codex.md) | The original project prompt, kept for context |
| [docs/qa-notes.md](docs/qa-notes.md) | Questions answered during the build, bugs hit, and verified reference numbers |

Where they disagree, that order is the order of authority.

## How this project is worked

- One milestone at a time, in order. No skipping ahead.
- Each task names the Python topics it needs. Those are **studied
  independently** before the task starts — the assistant lists them and says
  why, but does not teach them and does not set practice exercises.
- The implementation code is typed by the learner. An assistant sets tasks,
  reviews submitted code, and gives staged hints (conceptual → structural →
  pseudocode) after a genuine attempt. Full solutions only on explicit request.
- Each milestone is verified against explicit expected behavior and test cases
  before the next one begins.
- Files are created only when the current milestone needs them, so the tree
  stays small and every file has a reason to exist.

## Current status

**R1–R8 complete. Next up: R9 — optimizer progression (Momentum, RMSprop, Adam).**

| Milestone | Result |
| --- | --- |
| R1 Data and forward prediction | `y_hat = mx + c` over parallel collections |
| R2 Prediction error and MSE | MSE by hand, with a length guard |
| R3 Analytical gradients | `dL/dm`, `dL/dc` derived and verified by finite differences |
| R4 Gradient descent | training loop; four learning rates, one divergent |
| R5 Visualization | fitted line, loss curves, learning-rate comparison |
| R6 Noisy data | NumPy rewrite, seeded noise, 80/20 split, train vs test MSE |
| R7 Regularization | L1 and L2 by hand, swept across penalty strengths |
| R8 Batching | full-batch / mini-batch / SGD; only full-batch truly converges |

Recovering the true parameters from 50 noisy points, `lr=0.01`, 20,000 iterations:

```
learned     m=3.0707  c=6.2686      (true 3.0, 7.0)
train MSE   2.4766
test  MSE   1.6489
noise floor 4.0000   (sigma^2)
```

Files: `regression_from_scratch.py` (list-based functions), `regression_loop.py`
(training loop), `regression_plot.py` (R5 charts), `training.py` (NumPy functions
and seeded noisy data), `train_r6.py` (R6 training run and charts),
`train_r7.py` (self-contained regularization sweep), `train_r8.py`
(batching sweep, carrying the penalties forward).

## Environment

R1–R4 use only the Python standard library — nothing to install.

From R5 (NumPy) onward, use a virtualenv in this repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Confirm the interpreter is the one you expect (`which python`) before
installing anything. `requirements.txt` is created at R5 and grows as later
milestones need packages; it is deliberately not pre-populated.

## Eventual layout

Built up gradually, not created in advance:

```
RegLLM/
├── regression/
│   ├── regression_from_scratch.py
│   ├── optimizers.py
│   └── experiments/
├── tiny_llm/
│   ├── data/
│   ├── tokenizer.py
│   ├── embeddings.py
│   ├── attention.py
│   ├── transformer.py
│   ├── model.py
│   ├── train.py
│   ├── generate.py
│   └── evaluate.py
├── tests/
└── requirements.txt
```

## Tests

`pytest`, with tests in `tests/` named `test_<module>.py`. Introduced at R3,
where analytical gradients first need a finite-difference check — R1 and R2 are
verified against the printed-output test cases in [TASKS.md](TASKS.md). Numerical
tests state their tolerance; anything random fixes its seed.
