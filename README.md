# 🧠 Solomonoff-Inspired Hypothesis Ranking for LLMs

Implementation of "Solomonoff-Inspired Hypothesis Ranking with LLMs for Prediction Under Uncertainty"

**Authors:** Josh Barber, Rourke Young, Cameron Coombe, and Will Browne — Queensland University of Technology (QUT), 2025

---

## 📘 Overview

This repository contains an executable Python implementation of the Solomonoff-inspired hypothesis ranking framework described in the accompanying paper.

The script `testing.py` performs multi-hypothesis reasoning on Mini-ARC visual reasoning tasks using LLM-generated hypotheses that are scored and combined according to simplicity and predictive accuracy.

The goal is to approximate **Solomonoff induction** — a universal method for reasoning under uncertainty — in a practical, computable way using GPT-based hypothesis generation.

### 📄 Full Report

The complete research report is available in:
- **`28-09 -- Solomonoff Method - Josh Barber.pdf`**

---

## ⚙️ Method Summary

The system approximates Solomonoff induction by:

1. **Generating candidate hypotheses** using GPT-4, based on serialized object representations extracted from Mini-ARC input–output pairs.

2. **Scoring each hypothesis** according to:
   - **Simplicity**: shorter hypotheses (measured by token length) are preferred.
   - **Accuracy**: cell-wise predictive fit on the training examples.

3. **Combining the two multiplicatively** into a Solomonoff-inspired score:
   ```
   S(h) = Simplicity(h) × Accuracy(h)
   ```

4. **Normalising these scores** to form a discrete Solomonoff-weighted distribution over all hypotheses.

5. **Aggregating predictions** into a per-cell weighted matrix, representing uncertainty across hypotheses.

6. **Producing final outputs** via:
   - Leave-one-out evaluation,
   - Cell-wise probability distributions,
   - Weighted (argmax) grid predictions,
   - Markdown summaries with rankings and confidence analysis.

This process yields **uncertainty-aware predictions** robust to noisy or incomplete hypotheses, contrasting with traditional Bayesian Model Averaging (BMA) approaches that may over-concentrate confidence.

---

## 🧩 File: `testing.py`

### Key Components

| Section | Purpose |
|---------|---------|
| `LLMClient` | Interface for GPT-4 hypothesis generation with caching support. |
| `DiskCache` | JSON-based local caching of GPT responses. |
| `extract_json_from_text()` | Robust extraction of JSON from model responses. |
| `serialize_objects_for_gpt()` | Converts Mini-ARC grids into structured object representations. |
| `generate_hypotheses_gpt()` | Generates diverse candidate hypotheses from training examples. |
| `evaluate_hypothesis_on_problem()` | Evaluates hypotheses by cell-level accuracy on training data. |
| `simplicity_score()` | Approximates Solomonoff's length prior via token count. |
| `build_weighted_matrix()` | Constructs per-cell Solomonoff-weighted mixture probabilities. |
| `run()` | Main driver performing leave-one-out evaluation and report generation. |

---

## 🧮 Solomonoff Weighting Pipeline

```
Input Mini-ARC problem (JSON)
         ↓
Extract objects and spatial relations
         ↓
Generate hypotheses (GPT-4)
         ↓
Evaluate accuracy + simplicity
         ↓
Compute Solomonoff-inspired scores
         ↓
Apply hypotheses to held-out example
         ↓
Aggregate results into weighted matrix
         ↓
Export analysis (JSON + Markdown)
```

Each stage corresponds directly to Sections 3–4 of the paper.

---

## 📊 Outputs

The `run()` function produces the following outputs in a problem-specific directory inside the `out_folder`:

| File | Description |
|------|-------------|
| `full_leave_one_out_analysis.json` | Full evaluation results including all hypotheses, accuracies, and per-example predictions. |
| `solomonoff_eval_argmax.json` | Final grid predictions (argmax over weighted matrix). |
| `leave_one_out_summary.md` | Human-readable Markdown summary of ranked hypotheses, per-cell probabilities, and uncertainty visualisation. |
| `.llm_cache/` | Cached GPT responses for deterministic reproducibility. |

---

## 📁 Repository Structure

### Results Folders

The following directories contain experimental results referenced in the report:

- **`1/`, `2/`, `3/`, `4/`** — Results from different experimental runs using the Solomonoff method
- **`50MEGA/`, `60MEGA/`** — Large-scale experimental results
- **`bma/`** — Results using the Bayesian Model Averaging (BMA) method (a conversion of the Solomonoff method to BMA)

### Plotting Scripts

Python files with matching names are used to generate plots and visualizations for the methods described in the report. These scripts analyze and visualize the results from the corresponding results folders.

---

## 🧠 Example Usage

```python
if __name__ == '__main__':
    run(
        problem_file=r'MINI-ARC\data\MiniARC\connect_the_dots_to_make_a_square_l6aescjwslh5v45v3gf.json',
        out_folder='results',
        n_hyp=6,
        model='gpt-4'
    )
```

### Example CLI Execution

```bash
python testing.py
```

---

## 🧰 Requirements

- Python 3.9+
- OpenAI Python SDK (`openai` or modern openai client used in the code)
- `tiktoken` (optional, used to compute token-based simplicity)
- `dsl` (ARC DSL for object extraction — e.g. [ARC-DSL](https://github.com/arc-community/arc-dsl))
- Standard libs: `json`, `re`, `time`, `hashlib`, `pathlib`, `logging`, `argparse`

### Notes:

- **IMPORTANT:** Both `testing.py` and `bma.py` require an OpenAI API key. You must add your key to the `api_key` variable in each file:
  ```python
  api_key = "your-openai-api-key-here"
  ```
- If `tiktoken` is unavailable, the code degrades to a token-less simplicity approximation.

---

## 🧾 Theoretical Background

This implementation approximates **Solomonoff induction**, where:

```
P(x) = Σ_{p:U(p)=x*} 2^(-|p|)
```

but replaces infinite program enumeration with a finite pool of LLM-generated hypotheses. It preserves two core principles:

1. **Occam bias (simplicity prior)**: shorter descriptions are preferred via a token-length proxy.
2. **Data-driven posterior (likelihood)**: hypotheses that fit training data better receive higher weight via cell-wise accuracy.

See Sections 2.2–3.3 of the accompanying paper for full explanation and derivation.

---

## 🧩 Comparison to Bayesian Model Averaging (BMA)

| Aspect | Solomonoff Method | Bayesian Model Averaging |
|--------|-------------------|--------------------------|
| **Prior** | Simplicity-based (token length) | Uniform (in experiments) |
| **Likelihood** | Empirical accuracy (cell-wise) | Explicit per-cell likelihood with noise model |
| **Confidence** | Conservative / better calibrated under noisy hypotheses | Sharper; may be overconfident under noise |
| **Strength** | Robustness with noisy or incomplete hypotheses | Precision when hypothesis set is accurate |

### BMA Implementation

This repository includes a **Bayesian Model Averaging** implementation (`bma.py`) for comparison with the Solomonoff method. The BMA approach:

- Computes explicit per-cell categorical likelihoods P(D|M) with a noise parameter (ε)
- Uses likelihood-based posterior weights: `w_i ∝ P(D|M_i) × P(M_i)`
- Provides sharper, more concentrated predictions when hypotheses are accurate

**Key Difference:** While the Solomonoff method uses a simplicity prior (token length) combined with empirical accuracy, BMA uses a formal probabilistic likelihood model. The report compares both approaches across the Mini-ARC dataset to evaluate:
- Prediction accuracy
- Confidence calibration
- Robustness to hypothesis noise
- Computational efficiency

The `bma/` results folder contains outputs from the BMA method, enabling direct comparison with Solomonoff results in folders `1/`, `2/`, `3/`, `4/`, `50MEGA/`, and `60MEGA/`.

**Running BMA:**

BMA can be run in two modes:

1. **Generate fresh hypotheses** (independent BMA analysis):
```python
from bma import run
run(
    problem_file='MINI-ARC/data/MiniARC/problem.json',
    out_folder='bma_results',
    n_hyp=6,
    model='gpt-4'
)
```

2. **Convert existing Solomonoff results** (reuse hypotheses, apply BMA weighting):
```python
from bma import run_from_solomonoff_file
run_from_solomonoff_file(
    solomonoff_file='results/problem_id/full_leave_one_out_analysis.json',
    out_folder='bma_results'
)
```

The second approach is computationally efficient as it reuses GPT-generated hypotheses from the Solomonoff method and only recomputes the weights using BMA's probabilistic likelihood model.

---

## 🧪 Experimental Setup

- **Dataset**: Mini-ARC (a simplified ARC subset)
- **Model**: GPT-4 (zero-shot prompt-based hypothesis generation)
- **Evaluation**: Leave-one-out cross-validation (hold out one training example as eval)
- **Metrics**:
  - Cell-wise accuracy (strict cell-by-cell matching)
  - Simplicity (inverse token length normalized across hypotheses)
  - Composite Solomonoff-inspired score `S(h) = Simplicity × Accuracy`
  - Per-cell probability matrices aggregated from hypothesis weights

---

## 📈 Example Results (Illustrative)

The Markdown summary produced by `run()` ranks hypotheses and shows per-hypothesis performance. Example table entry:

| Rank | Hypothesis | Simplicity | Train Acc. | Solomonoff Score |
|------|-----------|------------|------------|------------------|
| 1 | Objects move toward grid centre, maintaining pattern | 0.87 | 0.92 | 0.80 |
| 2 | Cross clusters repositioned centrally | 0.85 | 0.91 | 0.77 |

The leave-one-out summary also shows per-cell top-2 probabilities and the final argmax grid.

---

## 📚 Citation

If you use this code, please cite the accompanying paper:

```
Barber, J., Young, R., Coombe, C., & Browne, W. (2025).
Solomonoff-Inspired Hypothesis Ranking with LLMs for Prediction Under Uncertainty.
Queensland University of Technology (QUT).
```

---

## ⚠️ Notes & Limitations

- The script relies on the quality of LLM-generated hypotheses; if the LLM fails to propose the correct transformation, no scoring scheme will recover it.
- This is a research prototype aimed at interpretability and reproducibility, not optimized for large-scale production use.
- Computational cost grows with `n_hyp` and the number/size of examples due to repeated LLM calls and evaluations. Use caching to reduce cost.
- Simplicity depends on the tokenizer (`tiktoken`) — different tokenisers change the simplicity ranking.

---

## 🧭 Future Extensions

Suggested improvements (discussed in the paper):

- Scale to full ARC / ARC-AGI datasets and run statistical evaluations.
- Replace token-length proxy with MDL-style compressors or more principled description-length metrics.
- Increase hypothesis diversity (different prompting strategies / multiple LLMs) and incorporate ensemble priors.
- Adapt the method to robotic planning tasks with structural uncertainty (e.g., pick-and-place strategies).

---

## 🏁 Summary

This code provides a computable Solomonoff induction approximation for LLM-driven reasoning under uncertainty. By combining a simplicity prior and empirical data fit, it produces calibrated, mixture-based predictions that explicitly represent per-cell uncertainty — useful in settings where multiple plausible explanations exist and we want to hedge rather than prematurely commit.

---

**Author:** Josh Barber  
**Supervisors:** Cameron Coombe, Will Browne  
**Affiliation:** Queensland University of Technology (QUT), Brisbane, Australia  
**Contact:** josh.barber@connect.qut.edu.au