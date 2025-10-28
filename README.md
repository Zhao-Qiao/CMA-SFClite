## CS599 M1 — GPT‑2 Gender Bias CMA and SFC‑lite Surgery (TransformerLens + SAE‑Lens)

**One‑liner**: This project reproduces and extends Vig et al. (2020) with Causal Mediation Analysis (CMA) to localize attention heads encoding occupation→gender pronoun stereotypes in GPT‑2, then applies a lightweight Sparse Feature Circuit (SFC‑lite) intervention using SAE‑Lens on `attn_out` (post‑W_O) features to sanity‑check debiasing effects.

---

### Table of Contents
- Background and Goals
- Project Layout
- Environment and Installation
- Quickstart
  - Run CMA (find biased heads)
  - Run SFC‑lite (gate SAE features driven by those heads)
- Methods
  - Bias metric and experimental setup
  - CMA details (pre‑W_O, per‑head)
  - SFC‑lite details (SAE + W_O row‑space attribution)
- Outputs and Visualization
- Reproducible Examples
- Runtime and Resource Tips
- FAQ
- References
- Status and Roadmap

---

### Background and Goals

- Paper context: Vig et al., 2020 “Investigating Gender Bias in Language Models Using Causal Mediation Analysis” studies gender pronoun stereotypes in Winogender‑style prompts.
- Goals of this repo:
  - Implement per‑head CMA with TransformerLens on GPT‑2 family to measure each attention head’s Natural Indirect Effect (NIE) on a fixed gender‑bias metric.
  - Use CMA’s Top‑K heads to drive an SFC‑lite intervention with SAE‑Lens on `attn_out` (post‑W_O): attribute SAE features to selected heads via W_O row‑space projectors, gate strongly attributed features, and observe bias changes.

---

### Project Layout

- `cma_gender_bias.py`
  - TransformerLens implementation of per‑head mediation at `blocks.L.attn.hook_z` (pre‑W_O head output `z`).
  - Fixed bias metric: next‑token `logit(" she") − logit(" he")` (neither token appears in the prompt).
  - Saves CSV (with rank/|NIE| and TE/NIE/NDE headers), prints Top‑K heads, optional Plotly heatmap.

- `sfc_lite_from_cma.py`
  - Reads the CMA CSV, selects Top‑K heads (optionally limiting unique layers).
  - Loads SAE‑Lens `attn_out` decoders per layer (default `gpt2-small-attn-out-v5-32k`).
  - Builds W_O row‑space projectors per head, attributes SAE decoder features to selected heads by projection mass, then gates features via a hybrid policy (Top‑N per layer + loose floor `tau_min` + optional global budget).
  - Prints a tiny bias sanity check before/after; saves plan to `sfc_surgery_plan.json`.

- `results/`
  - Stores CMA outputs (`*.csv`) and, if Plotly is installed, interactive heatmaps.

---

### Environment and Installation

Python ≥ 3.10 recommended. GPU is strongly recommended for medium/large models.

```bash
pip install -U torch transformer_lens sae_lens pandas plotly
```

Notes:
- First runs will download model and SAE weights (internet required).
- On GPU, install a `torch` build matching your CUDA. CPU works for small models but is slower.

---

### Quickstart

#### 1) Run CMA (find biased attention heads)

```bash
# Default: gpt2-small (alias of HF "gpt2")
python cma_gender_bias.py \
  --model gpt2-small \
  --occupation nurse \
  --gender-word man \
  --top-k 5 \
  --output-dir results

# Faster: distilgpt2; Higher fidelity: gpt2-medium / gpt2-large (more time/memory)
python cma_gender_bias.py --model distilgpt2
python cma_gender_bias.py --model gpt2-medium
```

Outputs:
- Console: baseline bias for base/counterfactual prompts, TE, per‑layer×head NIEs, Top‑K heads.
- CSV: `results/{model}_{occupation}_{gender_word}_{timestamp}.csv` with columns `rank, layer, head, nie, abs_nie` and header lines for TE/NIE/NDE.
- Heatmap (optional): interactive Plotly if available; otherwise a numeric printout.

#### 2) Run SFC‑lite (gate features driven by CMA heads)

```bash
python sfc_lite_from_cma.py \
  --cma_csv results/gpt2-small_nurse_man_20251028_042958.csv \
  --top_k_heads 5 \
  --topn_per_layer 300 \
  --tau_min 0.12 \
  --global_budget 3000

# Build the surgery plan only (no forward pass):
python sfc_lite_from_cma.py --cma_csv <path_to_csv> --dry_run
```

Outputs:
- Console: per‑layer gating counts/ratio, tiny bias sanity check (before/after), deltas.
- JSON: `sfc_surgery_plan.json` capturing SAE release, Top‑K, Top‑N, thresholds, budget, and per‑layer sizes.

---

### Methods

#### Bias metric and experimental setup

- Prompts:
  - base: `"The {occupation} said that"`
  - counterfactual: `"The {gender_word} said that"`
- Metric: next‑token `bias = logit(" she") − logit(" he")`.
- Effects:
  - `TE = bias(cf) − bias(base)`
  - Per‑head `NIE = bias_int − bias_clean` when replacing the mediator activation of one head from cf into base at the final position.
  - `NDE = TE − Σ NIE`

#### CMA details (`cma_gender_bias.py`)

- Mediator: `blocks.L.attn.hook_z` (pre‑W_O head output `z`).
- Loop over layers and heads: cache cf once per layer; on base, replace the final‑position `z` for that head with cf’s; forward to get `bias_int`; `NIE = bias_int − bias_clean`.
- Complexity: `n_layers × n_heads` interventions; GPU advised.

#### SFC‑lite details (`sfc_lite_from_cma.py`)

- Head selection: take Top‑K from the CMA CSV (prefer `rank`, else `abs_nie`, else `nie`), optionally limit the first N unique layers.
- SAE: load per‑layer `attn_out` SAE (default `gpt2-small-attn-out-v5-32k`).
- W_O row‑space projectors: build an orthogonal projector `P_h` for each head; project each SAE decoder vector `d_i` to get attribution mass `||P_h d_i||_2`; normalize per feature to obtain a mass distribution over heads.
- Hybrid selection policy:
  - Sum mass over selected heads, take Top‑N per layer (`--topn_per_layer`).
  - Drop features below a loose floor `--tau_min`.
  - Optionally enforce a global cap `--global_budget` across layers.
- Gating hook: at `blocks.L.hook_attn_out`, SAE encode → zero selected features → SAE decode to edited activations.

---

### Outputs and Visualization

- CMA CSV:
  - Header lines record TE/NIE/NDE and experiment description; rows include `rank, layer, head, nie, abs_nie`.
  - Use directly to pick Top‑K heads for surgery.

- Heatmap (optional):
  - X = head, Y = layer, color = NIE (red = increases female preference; blue = decreases).
  - High‑value cells (≥ 50% of max) are annotated.

- Surgery plan:
  - `sfc_surgery_plan.json` logs SAE release, selection hyper‑parameters, and per‑layer counts for reproducibility.

---

### Reproducible Examples

The `results/` folder already contains several CMA CSVs you can use with SFC‑lite:

```bash
python sfc_lite_from_cma.py \
  --cma_csv results/gpt2-small_nurse_man_20251028_042958.csv \
  --top_k_heads 5 --topn_per_layer 300 --tau_min 0.12 --global_budget 3000
```

Feel free to swap in other examples (e.g., `gpt2-small_doctor_woman_*.csv`, `gpt2-medium_teacher_person_*.csv`) to compare outcomes.

---

### Runtime and Resource Tips

Approximate CMA runtimes (with GPU):
- `distilgpt2`: ~2 min
- `gpt2-small`: ~3–5 min (recommended)
- `gpt2-medium`: ~10–15 min
- `gpt2-large`: ~30–45 min
- `gpt2-xl`: ~1–2 h

Tips:
- First‑time downloads will add overhead.
- If you hit VRAM limits:
  - Use a smaller model (`distilgpt2` / `gpt2-small`).
  - For SFC‑lite, lower `--top_k_heads`, `--topn_per_layer`, or set a smaller `--global_budget`.

---

### FAQ

- “transformer_lens/sae_lens not installed” errors:
  - Run `pip install -U transformer_lens sae_lens`.

- No Plotly; heatmap does not show:
  - Fallback printing is enabled; install Plotly for interactive figures.

- Windows path or permission issues:
  - Avoid special characters in the project path; running PowerShell as Administrator can help with write access.

- CUDA OOM:
  - Switch to a smaller model or CPU; reduce SFC‑lite selection sizes (see tips above).

---

### References

- Vig, J., Gehrmann, S., Belinkov, Y., Qian, S., Nevo, D., Singer, Y., & Shieber, S. (2020). Investigating Gender Bias in Language Models Using Causal Mediation Analysis.
- TransformerLens: [TransformerLens on GitHub](https://github.com/TransformerLensOrg/TransformerLens)
- SAE‑Lens: [SAE‑Lens on GitHub](https://github.com/jbloomAus/SAELens)

---

### Status and Roadmap (based on current progress)

Completed:
- CMA: pre‑W_O per‑head replacement; CSV and heatmap; Top‑K reporting.
- SFC‑lite: SAE feature attribution via W_O row‑space; gating; plan saved; bias sanity check.

Next steps:
- Evaluate on full Winogender and richer benchmarks (accuracy/calibration/diversity).
- Expand bias metrics (from logits difference to probabilities/log‑odds; more pronouns/occupations).
- Finer‑grained attribution (e.g., joint W_V/W_O subspaces or token‑conditioned analysis).
- Systematic comparison of SFC‑lite vs. full SFC and other debiasing methods in efficacy and cost.


