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
| [docs/qa-notes-regression.md](docs/qa-notes-regression.md) | Phase 1 Q&A: gradients, optimizers, bugs hit, verified reference numbers |
| [docs/qa-notes-llm.md](docs/qa-notes-llm.md) | Phase 2 Q&A: tokenization, attention, training the tiny LLM |

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

**Phase 1 complete (R1–R10). L1–L5 done. Next up: L6 — the transformer block.**

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
| R9 Optimizers | Momentum, RMSprop, Adam built by hand, with state traces |
| R10 Library comparison | sklearn agrees to 14 digits; Ridge/Lasso scaling worked out |

Recovering the true parameters from 50 noisy points, `lr=0.01`, 20,000 iterations:

```
learned     m=3.0707  c=6.2686      (true 3.0, 7.0)
train MSE   2.4766
test  MSE   1.6489
noise floor 4.0000   (sigma^2)
```

**`regression/`** — `regression_from_scratch.py` (list-based functions),
`regression_loop.py` (training loop), `regression_plot.py` (R5 charts),
`training.py` (NumPy functions and seeded noisy data), `train_r6.py` (noisy-data
run), `train_r7.py` (regularization sweep), `train_r8.py` (batching sweep),
`train_r9.py` (four optimizers as classes), `train_r10.py` (scikit-learn
comparison).

**`tiny_llm/`** — `data/kingdom.txt` (the eight-sentence corpus),
`tokenizer.py` (vocabulary and the token/ID mappings), `dataset.py`
(encode/decode and the 38 shifted training pairs), `embeddings.py` (token and
position tables, built in plain Python then PyTorch and checked against each
other), `attention_plain.py` (one causal head as plain loops),
`attention.py` (the same head in tensors, verified against the loops - this is
the one later milestones import), `multihead.py` (several heads plus an output
projection, explicit and fused versions checked against each other).

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
├── regression/            done - one script per milestone, mostly standalone
├── tiny_llm/
│   ├── data/kingdom.txt   L1
│   ├── tokenizer.py       L1  - shared, imported by everything below
│   ├── dataset.py         L2  - encode/decode, shifted sequences
│   ├── embeddings.py      L3
│   ├── attention_plain.py L4  - the loops, kept as the reference
│   ├── attention.py       L4
│   ├── multihead.py       L5
│   ├── transformer.py     L6
│   ├── model.py           L7
│   ├── train.py           L8, L9
│   ├── generate.py        L10
│   └── evaluate.py        L11
└── requirements.txt
```

Unlike `regression/`, where each milestone got a self-contained script,
`tiny_llm/` holds **shared modules** — `tokenizer.py` is imported by every
milestone from L2 onward, so duplicating it would invite the kind of drift that
let the R3 sign error survive three review rounds.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

`pytest`, in `tests/`, named `test_<module>.py`. `conftest.py` puts `regression/`
and `tiny_llm/` on `sys.path` so the modules' bare imports resolve from the repo
root.

Two files:

- **`test_regression.py`** — the reference values every Phase 1 milestone was
  checked against: `274.0` and `(-108.0, -32.0)` on the five clean points, the
  gradient sign table, analytical vs finite-difference agreement, the seeded
  dataset's shape and split, and the R6 result. Includes regression tests for the
  two real bugs — a function appending to a list it did not own, and NumPy
  gradients returning arrays because the reduction was missing.
- **`test_tokenizer.py`** — vocabulary contents and size, sorting, the ID round
  trip, contiguous IDs from zero, module-relative data path, and the 34.1%
  baseline.

The suite arrived at the Phase 1/2 boundary rather than at R3. Phase 1's scripts
were standalone, so breaking one could not break another, and verification lived
in each script's `__main__` block. `tiny_llm/` is shared modules — `tokenizer.py`
is imported by nine later files — so a change there can break something several
files away with nothing to notice. That is what the suite is for.
