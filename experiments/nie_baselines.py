"""
NIE Baselines Experiment
------------------------
Implements head-off and random feature ablation baselines using CMA-selected mediators.

This script now uses TransformerLens for activation tracing in place of nnsight while
keeping the original experimental logic intact.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import namedtuple
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer
from tqdm import tqdm


# ============================================================================
# Configuration
# ============================================================================

# Set to "resid" or "attn". Change this single line to switch SAE type.
SAE_MODE = "attn"

if SAE_MODE == "attn":
    SAE_RELEASE = "gpt2-small-attn-out-v5-128k"
    SAE_HOOK_TEMPLATE = "blocks.{layer}.hook_attn_out"
else:
    SAE_RELEASE = "gpt2-small-res-jb"
    SAE_HOOK_TEMPLATE = "blocks.{layer}.hook_resid_pre"


# ============================================================================
# Helper data structures
# ============================================================================


@dataclass(frozen=True)
class Submodule:
    """Hook point descriptor for TransformerLens."""

    name: str
    hook_name: str
    kind: str
    layer: int

    def __hash__(self) -> int:
        return hash((self.name, self.hook_name))


@dataclass
class NodeMask:
    """Boolean masks specifying which sparse features/residual coordinates to keep."""

    act: Optional[torch.Tensor] = None
    resc: Optional[torch.Tensor] = None

    def complemented(self) -> "NodeMask":
        act = None if self.act is None else (~self.act)
        resc = None if self.resc is None else (~self.resc)
        return NodeMask(act=act, resc=resc)


@dataclass
class PatchState:
    """Sparse feature decomposition for a mediator at a given prompt."""

    act: torch.Tensor
    res: Optional[torch.Tensor] = None

    def clone(self) -> "PatchState":
        return PatchState(
            act=self.act.clone(),
            res=None if self.res is None else self.res.clone(),
        )


DictionaryStash = namedtuple("DictionaryStash", ["embed", "attns", "mlps", "resids"])


# ============================================================================
# SAE loading utilities
# ============================================================================


def load_layer_sae(layer_idx: int, device: str) -> SAE:
    sae_id = SAE_HOOK_TEMPLATE.format(layer=layer_idx)
    sae = SAE.from_pretrained(
        release=SAE_RELEASE,
        sae_id=sae_id,
        device=device,
    )
    sae.eval()
    return sae


def load_saes_and_submodules(
    model: HookedTransformer,
    device: str,
) -> Tuple[DictionaryStash, Dict[Submodule, SAE]]:
    attns: List[Submodule] = []
    resids: List[Submodule] = []
    dictionaries: Dict[Submodule, SAE] = {}

    for layer_idx in range(model.cfg.n_layers):
        sae = load_layer_sae(layer_idx, device=device)
        hook_name = sae.cfg.metadata.get("hook_name", SAE_HOOK_TEMPLATE.format(layer=layer_idx))
        submodule = Submodule(
            name=f"{SAE_MODE}_{layer_idx}",
            hook_name=hook_name,
            kind=SAE_MODE,
            layer=layer_idx,
        )
        if SAE_MODE == "attn":
            attns.append(submodule)
        else:
            resids.append(submodule)
        dictionaries[submodule] = sae

    stash = DictionaryStash(embed=None, attns=attns, mlps=[], resids=resids)
    return stash, dictionaries


def find_mediator_submodule(
    mediator: Dict,
    stash: DictionaryStash,
) -> Tuple[Optional[Submodule], str]:
    layer_idx = mediator["layer"]
    mediator_type = mediator.get("type", "attn")

    if mediator_type == "attn" and layer_idx < len(stash.attns):
        return stash.attns[layer_idx], "attn"
    if mediator_type == "resid" and layer_idx < len(stash.resids):
        return stash.resids[layer_idx], "resid"
    if layer_idx < len(stash.resids):
        return stash.resids[layer_idx], "resid"
    return None, mediator_type


# ============================================================================
# TransformerLens helpers
# ============================================================================


def tokenize_prompt(model: HookedTransformer, prompt: str) -> torch.Tensor:
    tokens = model.to_tokens(prompt, prepend_bos=False)
    return tokens.to(model.cfg.device)


def _ensure_same_shape(name: str, a: torch.Tensor, b: torch.Tensor) -> None:
    if a.shape != b.shape:
        raise ValueError(f"{name} shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")


def _expand_to(tensor: torch.Tensor, target_ndim: int) -> torch.Tensor:
    expanded = tensor
    while expanded.dim() < target_ndim:
        expanded = expanded.unsqueeze(0)
    return expanded


def build_feature_edit_hook(
    dictionary: SAE,
    node_mask: NodeMask,
    patch_state: PatchState,
    handle_errors: str,
) -> callable:
    feature_dim = int(dictionary.cfg.d_sae)
    act_mask = None if node_mask.act is None else node_mask.act.to(dtype=torch.bool).reshape(-1)
    if act_mask is None:
        act_mask = torch.zeros(feature_dim, dtype=torch.bool)
    elif act_mask.numel() != feature_dim:
        raise ValueError("Feature mask size mismatch.")
    replace_mask = (~act_mask).to(dtype=torch.bool)

    resc_mask = None
    if node_mask.resc is not None:
        resc_mask = node_mask.resc.to(dtype=torch.bool).reshape(-1)

    def hook_fn(value: torch.Tensor, hook) -> torch.Tensor:
        features = dictionary.encode(value)
        recon = dictionary.decode(features)
        residual = value - recon

        patch_act = patch_state.act.to(value.device, value.dtype)
        _ensure_same_shape("feature", features, patch_act)

        mask_expanded = _expand_to(replace_mask.to(value.device), features.dim())
        features = torch.where(mask_expanded, patch_act, features)

        if patch_state.res is not None:
            patch_res = patch_state.res.to(value.device, value.dtype)
            _ensure_same_shape("residual", residual, patch_res)
            if resc_mask is not None:
                if resc_mask.numel() == 1:
                    if not resc_mask.item():
                        residual = patch_res
                else:
                    res_replace = _expand_to((~resc_mask.to(value.device)), residual.dim())
                    residual = torch.where(res_replace, patch_res, residual)

        return dictionary.decode(features) + residual

    return hook_fn


def sae_device(sae: SAE) -> torch.device:
    return next(sae.parameters()).device


def make_scaling_hook(
    dictionary: SAE,
    feature_indices: Optional[List[int]],
    alpha: float,
):
    device = sae_device(dictionary)
    feature_tensor = None
    if feature_indices is not None and feature_indices:
        feature_tensor = torch.tensor(feature_indices, device=device, dtype=torch.long)

    def hook_fn(value: torch.Tensor, hook) -> torch.Tensor:
        features = dictionary.encode(value)
        recon = dictionary.decode(features)
        residual = value - recon
        if feature_tensor is None:
            features = features * alpha
        else:
            features[..., feature_tensor] = features[..., feature_tensor] * alpha
        return dictionary.decode(features) + residual

    return hook_fn


def tokenize_corpus(
    model: HookedTransformer,
    path: str,
    max_corpus_tokens: Optional[int] = None,
) -> Optional[torch.Tensor]:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    tokens = model.to_tokens(text, prepend_bos=False)
    if tokens.numel() <= 1:
        return None
    if max_corpus_tokens is not None and tokens.size(1) > max_corpus_tokens:
        tokens = tokens[:, :max_corpus_tokens]
    return tokens


def compute_corpus_perplexity(
    model: HookedTransformer,
    tokens: torch.Tensor,
    hooks: Optional[List[Tuple[str, callable]]] = None,
    block_size: int = 512,
) -> float:
    if tokens is None or tokens.numel() <= 1:
        return float("nan")

    device = tokens.device
    total_log_prob = 0.0
    total_tokens = 0
    for start in range(0, tokens.size(1) - 1, block_size):
        end = min(start + block_size + 1, tokens.size(1))
        slice_tokens = tokens[:, start:end]
        if hooks:
            logits = model.run_with_hooks(
                slice_tokens,
                fwd_hooks=hooks,
                return_type="logits",
            )
        else:
            logits = model(slice_tokens, return_type="logits")
        log_probs = torch.nn.functional.log_softmax(logits[:, :-1], dim=-1)
        target = slice_tokens[:, 1:]
        gathered = torch.gather(log_probs, 2, target.unsqueeze(-1)).squeeze(-1)
        total_log_prob += gathered.sum().item()
        total_tokens += target.numel()

    if total_tokens == 0:
        return float("nan")
    avg_log_prob = total_log_prob / total_tokens
    return float(np.exp(-avg_log_prob))


def run_with_ablations(
    model: HookedTransformer,
    clean_prompt: str,
    patch_prompt: Optional[str],
    submodules: Sequence[Submodule],
    dictionaries: Dict[Submodule, SAE],
    nodes: Dict[Submodule, NodeMask],
    complement: bool = False,
    ablation_fn=None,
    handle_errors: str = "default",
) -> torch.Tensor:
    if not submodules:
        raise ValueError("No submodules provided for ablation.")

    if patch_prompt is None:
        patch_prompt = clean_prompt

    clean_tokens = tokenize_prompt(model, clean_prompt)
    patch_tokens = tokenize_prompt(model, patch_prompt)

    hook_names = {submodule.hook_name for submodule in submodules}

    with torch.no_grad():
        _, patch_cache = model.run_with_cache(
            patch_tokens,
            names_filter=lambda name: name in hook_names,
            remove_batch_dim=False,
            return_type="logits",
        )

    patch_states: Dict[Submodule, PatchState] = {}
    for submodule in submodules:
        dictionary = dictionaries[submodule]
        activation = patch_cache[submodule.hook_name]
        features = dictionary.encode(activation)
        recon = dictionary.decode(features)
        residual = activation - recon
        state = PatchState(act=features, res=residual)
        if ablation_fn is not None:
            state = ablation_fn(state)
        patch_states[submodule] = state

    hooks = []
    for submodule in submodules:
        dictionary = dictionaries[submodule]
        node_mask = nodes[submodule]
        if complement:
            node_mask = node_mask.complemented()
        hook_fn = build_feature_edit_hook(
            dictionary=dictionary,
            node_mask=node_mask,
            patch_state=patch_states[submodule],
            handle_errors=handle_errors,
        )
        hooks.append((submodule.hook_name, hook_fn))

    with torch.no_grad():
        logits = model.run_with_hooks(
            clean_tokens,
            fwd_hooks=hooks,
            return_type="logits",
        )
    return logits


# ============================================================================
# Metric helpers
# ============================================================================


def compute_bias_and_ppl(model: HookedTransformer, logits: torch.Tensor, tokens: torch.Tensor) -> Tuple[float, float]:
    id_she = model.tokenizer(" she", add_special_tokens=False)["input_ids"][0]
    id_he = model.tokenizer(" he", add_special_tokens=False)["input_ids"][0]
    bias = (logits[0, -1, id_she] - logits[0, -1, id_he]).item()
    ppl = _perplexity_from_logits(logits[0], tokens)
    return bias, ppl


def _perplexity_from_logits(logits_val: torch.Tensor, tokens: torch.Tensor) -> float:
    if len(tokens) < 2:
        return float("inf")
    log_probs = torch.nn.functional.log_softmax(logits_val[:-1], dim=-1)
    target_log_probs = log_probs[range(len(tokens) - 1), tokens[1:]]
    return torch.exp(-target_log_probs.mean()).item()


def determine_dictionary_size(
    model: HookedTransformer,
    submodule: Submodule,
    dictionary: SAE,
    prompt: str,
) -> int:
    return int(dictionary.cfg.d_sae)


def apply_alpha_gate(
    model: HookedTransformer,
    mediator: Dict,
    submodule: Submodule,
    dictionary: SAE,
    base_prompt: str,
    cf_prompt: str,
    feature_indices: Optional[List[int]],
    alpha: float,
) -> Tuple[float, float, float, float, float]:
    tokens_base = tokenize_prompt(model, base_prompt)[0]

    dict_size = int(dictionary.cfg.d_sae)
    mask_keep = torch.ones(dict_size, dtype=torch.bool)
    if feature_indices is None:
        mask_keep[:] = False
    else:
        mask_keep[feature_indices] = False

    def make_nodes() -> Dict[Submodule, NodeMask]:
        return {
            submodule: NodeMask(
                act=mask_keep.clone(),
                resc=torch.ones(1, dtype=torch.bool),
            )
        }

    def ablation_fn(state: PatchState) -> PatchState:
        act = state.act.clone()
        target_mask = (~mask_keep).to(torch.bool)
        if feature_indices is None:
            act = act * alpha
        else:
            act[..., target_mask] = act[..., target_mask] * alpha
        res = state.res.clone() if state.res is not None else None
        return PatchState(act=act, res=res)

    with torch.no_grad():
        logits_clean = model(tokenize_prompt(model, base_prompt), return_type="logits")

    logits_gated = run_with_ablations(
        model=model,
        clean_prompt=base_prompt,
        patch_prompt=base_prompt,
        submodules=[submodule],
        dictionaries={submodule: dictionary},
        nodes=make_nodes(),
        ablation_fn=ablation_fn,
        handle_errors="keep",
    )

    logits_cf = run_with_ablations(
        model=model,
        clean_prompt=base_prompt,
        patch_prompt=cf_prompt,
        submodules=[submodule],
        dictionaries={submodule: dictionary},
        nodes=make_nodes(),
        ablation_fn=ablation_fn,
        handle_errors="keep",
    )

    bias_clean, ppl_clean = compute_bias_and_ppl(model, logits_clean, tokens_base)
    bias_gated, ppl_gated = compute_bias_and_ppl(model, logits_gated, tokens_base)
    bias_cf_replaced, _ = compute_bias_and_ppl(model, logits_cf, tokens_base)
    nie = bias_cf_replaced - bias_gated
    return bias_clean, bias_gated, ppl_clean, ppl_gated, nie


# ============================================================================
# Baseline experiment logic
# ============================================================================


def run_baselines(
    model_name: str,
    mediator_path: str,
    eval_data_path: str,
    output_path: str,
    topk: int,
    control_count: int,
    num_features: int,
    seed: int,
    device: str,
    corpus_path: Optional[str],
    max_corpus_tokens: Optional[int],
) -> str:
    if SAE_MODE not in {"resid", "attn"}:
        raise ValueError("SAE_MODE must be 'resid' or 'attn'")

    print("=" * 70)
    print("NIE Baselines")
    print("=" * 70)
    print(f"模型: {model_name}")
    print(f"Top-K Mediators 文件: {mediator_path}")
    print(f"评估数据: {eval_data_path}")
    print(f"SAE 模式: {SAE_MODE}")
    print(f"Top-K: {topk}, 控制数量: {control_count}")
    print("=" * 70)

    dtype = torch.float32
    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        dtype=dtype,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
    )
    model.eval()

    stash, dictionaries = load_saes_and_submodules(model, device=device)

    with open(mediator_path, "r", encoding="utf-8") as f:
        mediator_data = json.load(f)
    mediators = mediator_data if isinstance(mediator_data, list) else mediator_data.get("mediators", [])

    total_needed = topk + control_count
    mediators = mediators[:total_needed] if total_needed > 0 else mediators
    mediators_main = mediators[:topk]
    mediators_control = mediators[topk:topk + control_count]

    mediator_entries = [(mediator, "topk") for mediator in mediators_main]
    mediator_entries.extend((mediator, "control") for mediator in mediators_control)

    with open(eval_data_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    examples = eval_data.get("examples", eval_data if isinstance(eval_data, list) else [])
    if not examples:
        raise ValueError("评估数据为空")

    rng = np.random.default_rng(seed)
    results: List[Dict] = []

    corpus_tokens = tokenize_corpus(model, corpus_path, max_corpus_tokens=max_corpus_tokens)
    baseline_corpus_ppl = compute_corpus_perplexity(model, corpus_tokens) if corpus_tokens is not None else float("nan")

    for idx, (mediator, category) in enumerate(tqdm(mediator_entries, desc="Mediators")):
        submodule, mediator_type = find_mediator_submodule(mediator, stash)
        if submodule is None:
            print(f"  ⚠ Mediator {idx} 无法找到对应 hook，跳过")
            continue
        dictionary = dictionaries[submodule]

        dict_size = int(dictionary.cfg.d_sae)

        scenarios = [
            {
                "label": "head_off",
                "feature_indices": None,
                "alpha": 0.0,
                "num_features": 1,
                "feature_source": "cma" if category == "topk" else category,
            }
        ]

        if dict_size > 0:
            rand_features = rng.choice(dict_size, size=min(num_features, dict_size), replace=False).tolist()
            scenarios.append({
                "label": "random_cut",
                "feature_indices": rand_features,
                "alpha": 0.0,
                "num_features": len(rand_features),
                "feature_source": "random",
            })

        for scenario in scenarios:
            bias_clean_vals = []
            bias_edit_vals = []
            ppl_clean_vals = []
            ppl_edit_vals = []
            nie_vals = []

            for example_idx, example in enumerate(examples):
                base_prompt = example.get("base", example.get("clean_prefix", ""))
                cf_prompt = example.get("counterfactual", example.get("patch_prefix", ""))
                if not base_prompt or not cf_prompt:
                    continue

                bias_base_orig, _, ppl_base_orig, _, nie_orig = apply_alpha_gate(
                    model=model,
                    mediator=mediator,
                    submodule=submodule,
                    dictionary=dictionary,
                    base_prompt=base_prompt,
                    cf_prompt=cf_prompt,
                    feature_indices=None,
                    alpha=1.0,
                )

                _, bias_base_gated, _, ppl_base_gated, nie_after = apply_alpha_gate(
                    model=model,
                    mediator=mediator,
                    submodule=submodule,
                    dictionary=dictionary,
                    base_prompt=base_prompt,
                    cf_prompt=cf_prompt,
                    feature_indices=scenario["feature_indices"],
                    alpha=scenario["alpha"],
                )

                bias_clean_vals.append(bias_base_orig)
                bias_edit_vals.append(bias_base_gated)
                ppl_clean_vals.append(ppl_base_orig)
                ppl_edit_vals.append(ppl_base_gated)
                nie_vals.append(nie_after - nie_orig)

            if not bias_clean_vals:
                continue

            mean_bias_clean = float(np.mean(bias_clean_vals))
            mean_bias_edit = float(np.mean(bias_edit_vals))
            mean_ppl_clean = float(np.mean(ppl_clean_vals))
            mean_ppl_edit = float(np.mean(ppl_edit_vals))
            mean_delta_nie = float(np.mean(nie_vals))

            remaining_bias_pct = 1.0
            if abs(mean_bias_clean) > 1e-9:
                remaining_bias_pct = float(abs(mean_bias_edit) / abs(mean_bias_clean))

            gated_corpus_ppl = float("nan")
            delta_corpus_ppl = float("nan")
            if corpus_tokens is not None:
                hook_fn = make_scaling_hook(dictionary, scenario["feature_indices"], scenario["alpha"])
                gated_corpus_ppl = compute_corpus_perplexity(
                    model,
                    corpus_tokens,
                    hooks=[(submodule.hook_name, hook_fn)],
                )
                if not np.isnan(baseline_corpus_ppl):
                    delta_corpus_ppl = gated_corpus_ppl - baseline_corpus_ppl

            aggregate_row = {
                "analysis": "baseline",
                "edit_label": scenario["label"],
                "feature_source": scenario["feature_source"],
                "feature_count": scenario["num_features"],
                "alpha": scenario["alpha"],
                "sum_abs_edit": abs(1.0 - scenario["alpha"]) * max(1, scenario["num_features"]),
                "bias_original_mean": mean_bias_clean,
                "bias_edited_mean": mean_bias_edit,
                "ppl_original_mean": mean_ppl_clean,
                "ppl_edited_mean": mean_ppl_edit,
                "delta_ppl_mean": mean_ppl_edit - mean_ppl_clean,
                "remaining_bias_pct": remaining_bias_pct,
                "delta_nie_mean": mean_delta_nie,
                "corpus_ppl_original": baseline_corpus_ppl,
                "corpus_ppl_edited": gated_corpus_ppl,
                "delta_corpus_ppl": delta_corpus_ppl,
                "mediator_layer": mediator["layer"],
                "mediator_type": mediator_type,
                "mediator_head": mediator.get("head"),
                "mediator_nie": mediator.get("nie"),
                "mediator_category": category,
                "row_type": "aggregate",
            }

            results.append(aggregate_row)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({key for row in results for key in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✓ 结果已写入 {output_path}")
    plot_baseline_results(results, output_path)
    return output_path


def plot_baseline_results(rows: List[Dict], csv_path: str) -> None:
    aggregate_rows = [r for r in rows if "bias_edited_mean" in r]
    if not aggregate_rows:
        return
    remaining = [r.get("remaining_bias_pct", 1.0) * 100 for r in aggregate_rows]
    delta_ppl = []
    for r in aggregate_rows:
        val = r.get("delta_corpus_ppl")
        if val is None or np.isnan(val):
            val = r.get("delta_ppl_mean", float("nan"))
        delta_ppl.append(val)
    labels = [r.get("edit_label", "edit") for r in aggregate_rows]

    plt.figure(figsize=(8, 5))
    for x, y, lbl in zip(remaining, delta_ppl, labels):
        plt.scatter(x, y, label=lbl, alpha=0.7)
    plt.xlabel("剩余偏差（% 基线）")
    plt.ylabel("ΔPPL（语料）")
    plt.title("Baseline Pareto (Bias vs ΔPPL)")
    handles, unique_labels = [], []
    for lbl in sorted(set(labels)):
        handles.append(plt.Line2D([0], [0], marker="o", color="w", label=lbl, markerfacecolor="tab:blue", markersize=8))
    if handles:
        plt.legend(handles, [h.get_label() for h in handles])
    plt.grid(True, alpha=0.3)
    plot_path = os.path.splitext(csv_path)[0] + ".png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ 绘制基线图: {plot_path}")


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NIE Baselines (Head-off, Random Cut)")
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--topk_json", type=str, default="/Users/qjzheng/Desktop/CMA-SFC-integration/experiments/data/topk_mediators_gpt2-small_doctor_woman_20251026_150239.json")
    parser.add_argument("--eval_data", type=str, default="/Users/qjzheng/Desktop/CMA-SFC-integration/experiments/data/bias_eval_gpt2-small_doctor_woman_20251026_150239.json")
    parser.add_argument("--output", type=str, default="results/nie_baselines.csv")
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--controls", type=int, default=3)
    parser.add_argument("--num_features", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--corpus_path", type=str, default="test.txt", help="用于 ΔPPL 计算的评估语料（可为空）")
    parser.add_argument("--max_corpus_tokens", type=int, default=4096, help="语料最大 token 数（防止过大）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_baselines(
        model_name=args.model,
        mediator_path=args.topk_json,
        eval_data_path=args.eval_data,
        output_path=args.output,
        topk=args.topk,
        control_count=args.controls,
        num_features=args.num_features,
        seed=args.seed,
        device=args.device,
        corpus_path=args.corpus_path,
        max_corpus_tokens=args.max_corpus_tokens,
    )


if __name__ == "__main__":
    main()
