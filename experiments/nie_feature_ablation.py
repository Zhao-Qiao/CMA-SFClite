"""NIE - Feature Ablation 实验（使用完整的 SFC 源码）

整合 CMA 和 SFC 方法，对 Top-K mediators 的稀疏特征进行消融实验

实验流程：
1. 读取 CMA 的 Top-K mediator 列表
2. 对每个 mediator：
   - 加载对应层的 SAE
   - 使用 SFC 的 patching_effect 计算特征效应
   - 使用 SFC 的 ablation 工具进行特征消融
   - 计算 NIE_f 和 ΔPPL_f
3. 可视化结果
"""

import argparse
import json
import os
from datetime import datetime
from typing import List, Dict, Optional
import csv
import math
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

try:
    from nnsight import LanguageModel
except ImportError:
    print("错误：需要安装 nnsight")
    print("请运行：pip install nnsight")
    exit(1)

# 导入本地模块
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cma_utils import load_topk_mediators
from sfc_utils import (
    load_saes_and_submodules,
    DictionaryStash,
    patching_effect,
    run_with_ablations,
    SparseAct,
)


def compute_perplexity(model, prompt: str) -> float:
    """计算给定 prompt 的困惑度"""
    with torch.no_grad():
        with model.trace(prompt):
            logits = model.output.logits.save()
        
        # 获取 token ids
        tokens = model.tokenizer(prompt, return_tensors='pt')['input_ids'][0]
        
        # 计算 cross-entropy loss
        logits_val = logits.value[0]  # (seq_len, vocab_size)
        
        # 只计算预测 token 的 loss（从第2个开始）
        if len(tokens) < 2:
            return float('inf')
        
        ppl = _perplexity_from_logits(logits_val, tokens)
        
    return ppl


def _perplexity_from_logits(logits_val: torch.Tensor, tokens: torch.Tensor) -> float:
    """根据 logits 和目标 token 计算困惑度"""
    if len(tokens) < 2:
        return float('inf')
    
    log_probs = torch.nn.functional.log_softmax(logits_val[:-1], dim=-1)
    target_log_probs = log_probs[range(len(tokens) - 1), tokens[1:]]
    ppl = torch.exp(-target_log_probs.mean()).item()
    return ppl


def find_mediator_submodule(mediator: Dict, submodules_stash: DictionaryStash) -> tuple:
    """
    根据 mediator 信息找到对应的 submodule
    
    返回：(submodule, component_type)
    """
    layer_idx = mediator['layer']
    mediator_type = mediator.get('type', 'attn')
    
    if mediator_type == 'attn':
        if layer_idx < len(submodules_stash.attns):
            return submodules_stash.attns[layer_idx], 'attn'
    elif mediator_type == 'mlp':
        if layer_idx < len(submodules_stash.mlps):
            return submodules_stash.mlps[layer_idx], 'mlp'
    elif mediator_type == 'resid':
        if layer_idx < len(submodules_stash.resids):
            return submodules_stash.resids[layer_idx], 'resid'
    
    return None, None


def ablate_feature_with_sfc(
    model,
    submodules_stash: DictionaryStash,
    dictionaries: Dict,
    base_prompt: str,
    cf_prompt: str,
    mediator: Dict,
    feature_indices: Optional[List[int]],
    alpha_values: List[float],
    edit_label: str,
) -> List[Dict]:
    layer_idx = mediator['layer']
    
    # 找到对应的 submodule
    submodule, comp_type = find_mediator_submodule(mediator, submodules_stash)
    
    if submodule is None:
        print(f"  警告：无法找到 layer {layer_idx} 的 submodule")
        return []
    
    dictionary = dictionaries[submodule]
    dict_size = getattr(dictionary, 'dict_size', None)
    if dict_size is None:
        if hasattr(dictionary, 'decoder') and hasattr(dictionary.decoder, 'weight'):
            dict_size = dictionary.decoder.weight.shape[-1]
        else:
            raise ValueError("无法确定 SAE 字典大小，终止消融。")

    if feature_indices is not None:
        feature_indices = [int(idx) for idx in feature_indices if int(idx) < dict_size]
        if not feature_indices:
            print("  警告：特征集合为空，跳过")
            return []

    # 获取原始 bias / NIE / PPL
    def trace_prompt(prompt: str):
        with torch.no_grad():
            with model.trace(prompt):
                logits = model.output.logits.save()
        return logits.value

    base_logits = trace_prompt(base_prompt)
    cf_logits = trace_prompt(cf_prompt)

    base_tokens = model.tokenizer(base_prompt, return_tensors='pt')['input_ids'][0]
    cf_tokens = model.tokenizer(cf_prompt, return_tensors='pt')['input_ids'][0]

    def bias_from_logits(logits_val: torch.Tensor) -> float:
        id_she = model.tokenizer(" she", add_special_tokens=False)["input_ids"][0]
        id_he = model.tokenizer(" he", add_special_tokens=False)["input_ids"][0]
        return (logits_val[0, -1, id_she] - logits_val[0, -1, id_he]).item()

    nie_original = bias_from_logits(cf_logits) - bias_from_logits(base_logits)
    ppl_original = _perplexity_from_logits(base_logits[0], base_tokens)

    mask_keep = torch.ones(dict_size, dtype=torch.bool)
    if feature_indices is None:
        mask_keep[:] = False
    else:
        mask_keep[feature_indices] = False

    def make_nodes(mask: torch.Tensor) -> Dict:
        return {
            submodule: SparseAct(
                act=mask.clone(),
                resc=torch.ones(1, dtype=torch.bool)
            )
        }

    target_mask = (~mask_keep).to(torch.bool)

    def make_ablation_fn(alpha: float):
        def ablation_fn(sparse_act: SparseAct) -> SparseAct:
            act = sparse_act.act.clone()
            res = sparse_act.res.clone() if sparse_act.res is not None else None
            if feature_indices is None:
                act = act * alpha
            else:
                act[..., target_mask] = act[..., target_mask] * alpha
            return SparseAct(act=act, res=res, resc=sparse_act.resc.clone() if sparse_act.resc is not None else None)
        return ablation_fn

    def run_gate(prompt: str, alpha: float) -> torch.Tensor:
        return run_with_ablations(
            clean=prompt,
            patch=prompt,
            model=model,
            submodules=[submodule],
            dictionaries=dictionaries,
            nodes=make_nodes(mask_keep),
            metric_fn=lambda m: m.output.logits,
            ablation_fn=make_ablation_fn(alpha),
            complement=False,
            handle_errors='keep',
        )

    results = []
    for alpha in sorted(set(alpha_values + [1.0]), reverse=True):
        base_logits_alpha = run_gate(base_prompt, alpha)
        cf_logits_alpha = run_gate(cf_prompt, alpha)

        bias_base_alpha = bias_from_logits(base_logits_alpha)
        bias_cf_alpha = bias_from_logits(cf_logits_alpha)
        nie_alpha = bias_cf_alpha - bias_base_alpha

        ppl_base_alpha = _perplexity_from_logits(base_logits_alpha[0], base_tokens)

        results.append({
            'alpha': float(alpha),
            'nie_original': float(nie_original),
            'nie_ablated': float(nie_alpha),
            'delta_nie': float(nie_alpha - nie_original),
            'ppl_original': float(ppl_original),
            'ppl_ablated': float(ppl_base_alpha),
            'delta_ppl': float(ppl_base_alpha - ppl_original),
            'edit_label': edit_label,
            'feature_idx': -1 if feature_indices is None else (feature_indices[0] if len(feature_indices) == 1 else -2),
        })

    return results


def run_experiment(
    model_name: str,
    topk_mediators_path: str,
    eval_data_path: str,
    output_dir: str,
    topk: int = 5,
    num_features_per_mediator: int = 10,
    alpha_values: Optional[List[float]] = None,
    max_examples: int = 3,
    include_random: bool = True,
    include_global: bool = True,
    include_feature_gating: bool = True,
    include_head_off: bool = True,
    control_mediators: int = 0,
    nie_thresholds: Optional[List[float]] = None,
    seed: int = 0,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
):
    """
    按照 proposal 运行闭环实验：
      1. 使用 CMA 的 Top-K mediators
      2. 用 SFC-lite 验证节点效应并应用软 gating（含离散 ablation）
      3. 评估 Bias–Perplexity Pareto，并记录单点编辑的 NIE vs ΔPPL
    """
    print("=" * 70)
    print("NIE - Feature Ablation 实验（SFC-lite 门控）")
    print("=" * 70)
    print(f"模型: {model_name}")
    print(f"Top-K Mediators: {topk_mediators_path}")
    print(f"评估数据: {eval_data_path}")
    print(f"Top-K: {topk}")
    print(f"特征数（每个 mediator）: {num_features_per_mediator}")
    print(f"软门控 alpha 列表: {alpha_values}")
    print(f"最大样本数: {max_examples if max_examples > 0 else '全部'}")
    print(f"包含随机基线: {include_random}")
    print(f"包含全子空间门控: {include_global}")
    print(f"包含特征级门控: {include_feature_gating}")
    print(f"包含 head-off 基线: {include_head_off}")
    print(f"低 NIE 控制 mediator 数: {control_mediators}")
    print(f"匹配 debiasing 阈值: {nie_thresholds}")
    print("=" * 70)

    if alpha_values is None or len(alpha_values) == 0:
        alpha_values = [1.0, 0.75, 0.5, 0.25, 0.0]

    rng = np.random.default_rng(seed)

    # 1. 加载模型
    print("\n[1/6] 加载模型...")
    if model_name in ["EleutherAI/pythia-70m-deduped", "gpt2", "gpt2-small"]:
        dtype = torch.float32
    else:
        dtype = torch.bfloat16

    model = LanguageModel(
        model_name,
        device_map=device,
        torch_dtype=dtype,
        dispatch=True,
    )
    print(f"  ✓ 模型已加载到 {device}")

    # 2. 加载 SAE 和 submodules
    print("\n[2/6] 加载 SAE 和 submodules...")
    try:
        submodules_stash, dictionaries = load_saes_and_submodules(
            model,
            separate_by_type=True,
            include_embed=False,
            dtype=dtype,
            device=device,
        )
        print(f"  ✓ 加载了 {len(submodules_stash.attns)} 层的 SAE")
    except Exception as e:
        print(f"  ✗ 加载 SAE 失败: {e}")
        print("  提示：请确保已下载 SAE 字典或设置正确的路径")
        return

    # 3. 加载 Top-K Mediators
    print("\n[3/6] 加载 Top-K Mediators...")
    requested_count = topk + control_mediators + 10
    if topk_mediators_path.endswith('.csv'):
        mediators_all = load_topk_mediators(topk_mediators_path, topk=requested_count)
    elif topk_mediators_path.endswith('.json'):
        from cma_utils.mediation import load_topk_mediators_from_json
        mediators_all = load_topk_mediators_from_json(topk_mediators_path, topk=requested_count)
    else:
        raise ValueError(f"不支持的文件格式: {topk_mediators_path}")
    if topk is not None and topk > 0:
        mediators_main = mediators_all[:topk]
    else:
        mediators_main = mediators_all

    control_list = []
    if control_mediators > 0 and len(mediators_all) > topk:
        control_candidates = mediators_all[topk:]
        control_list = control_candidates[-control_mediators:] if len(control_candidates) >= control_mediators else control_candidates

    mediator_entries = []
    seen_ids = set()

    def mediator_key(med: Dict) -> tuple:
        return (med.get('layer'), med.get('type'), med.get('head'), med.get('position'))

    for med in mediators_main:
        key = mediator_key(med)
        if key not in seen_ids:
            mediator_entries.append((med, 'topk'))
            seen_ids.add(key)

    for med in control_list:
        key = mediator_key(med)
        if key not in seen_ids:
            mediator_entries.append((med, 'control'))
            seen_ids.add(key)

    print(f"  ✓ 主 mediator: {len(mediators_main)} 个, 控制: {len(control_list)} 个, 总计 {len(mediator_entries)}")

    # 4. 加载评估数据
    print("\n[4/6] 加载评估数据...")
    with open(eval_data_path, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)
    examples = eval_data.get('examples', eval_data if isinstance(eval_data, list) else [])
    print(f"  ✓ 加载了 {len(examples)} 个评估样本")
    if len(examples) == 0:
        print("  ✗ 无样本可用，终止。")
        return
    if max_examples > 0:
        examples = examples[:max_examples]

    # 5. 运行 SFC-lite 门控实验
    print("\n[5/6] 运行软门控与消融实验...")
    all_results: List[Dict] = []

    def metric_fn(metric_model):
        logits = metric_model.output.logits[:, -1, :]
        id_she = metric_model.tokenizer(" she", add_special_tokens=False)["input_ids"][0]
        id_he = metric_model.tokenizer(" he", add_special_tokens=False)["input_ids"][0]
        return logits[:, id_she] - logits[:, id_he]

    for med_idx, (mediator, mediator_category) in enumerate(tqdm(mediator_entries, desc="Mediators")):
        layer_idx = mediator['layer']
        mediator_type = mediator.get('type', 'attn')
        mediator_head = mediator.get('head')
        print(f"\n  Mediator #{med_idx + 1}: Layer {layer_idx}, Type {mediator_type}, Head {mediator_head}, Category {mediator_category}")

        submodule, comp_type = find_mediator_submodule(mediator, submodules_stash)
        if submodule is None:
            print("    跳过：无法定位 submodule")
            continue

        dictionary = dictionaries[submodule]
        dict_size = getattr(dictionary, 'dict_size', None)
        if dict_size is None:
            if hasattr(dictionary, 'decoder') and hasattr(dictionary.decoder, 'weight'):
                dict_size = dictionary.decoder.weight.shape[-1]
            else:
                print("    跳过：SAE 字典缺少 dict_size 信息")
                continue

        # 5.1 节点效应验证（使用第一条样本的 patching_effect）
        ref_example = examples[0]
        base_prompt_ref = ref_example.get('base', ref_example.get('clean_prefix', ''))
        cf_prompt_ref = ref_example.get('counterfactual', ref_example.get('patch_prefix', ''))
        patch_total_effect = None
        feature_scores = torch.zeros(dict_size)
        if base_prompt_ref and cf_prompt_ref:
            try:
                effects, _, _, total_effect = patching_effect(
                    clean=base_prompt_ref,
                    patch=cf_prompt_ref,
                    model=model,
                    submodules=[submodule],
                    dictionaries=dictionaries,
                    metric_fn=metric_fn,
                    method='attrib',
                )
                patch_total_effect = total_effect.item() if total_effect is not None else None
                sub_effect = effects[submodule]
                if hasattr(sub_effect, 'act'):
                    feature_scores = sub_effect.act.abs().mean(dim=(0, 1)).detach().cpu()
            except Exception as exc:
                print(f"    警告：patching_effect 失败，原因: {exc}")
        else:
            print("    警告：参考样本缺少 base/cf prompt，跳过节点验证")

        topk_features = []
        if include_feature_gating and dict_size > 0:
            num_select = min(num_features_per_mediator, dict_size)
            if feature_scores.max() > 0:
                topk_features = torch.topk(feature_scores, k=num_select).indices.tolist()
            else:
                topk_features = list(range(num_select))

        random_features = []
        if include_random and dict_size > 0:
            num_random = len(topk_features) if topk_features else min(num_features_per_mediator, dict_size)
            available = np.setdiff1d(np.arange(dict_size), np.array(topk_features), assume_unique=True)
            if len(available) >= num_random:
                random_features = rng.choice(available, size=num_random, replace=False).tolist()
            else:
                random_features = rng.choice(np.arange(dict_size), size=num_random, replace=True).tolist()

        aggregate_buffer = {}

        def record_rows(rows: List[Dict], meta: Dict):
            for row in rows:
                combined = {**meta, **row}
                all_results.append(combined)

                key = (
                    combined['mediator_layer'],
                    combined['feature_idx'],
                    combined['alpha'],
                    combined['feature_source'],
                    combined['edit_label'],
                    combined['mediator_category'],
                )
                if key not in aggregate_buffer:
                    aggregate_buffer[key] = {'rows': [], 'meta': {k: combined[k] for k in meta_keys}}
                aggregate_buffer[key]['rows'].append(combined)

        meta_keys = [
            'mediator_layer',
            'mediator_type',
            'mediator_head',
            'mediator_nie',
            'patching_effect_nie',
             'mediator_category',
            'feature_idx',
            'feature_source',
            'edit_label',
            'feature_score',
            'alpha',
        ]

        for example_idx, example in enumerate(examples):
            base_prompt = example.get('base', example.get('clean_prefix', ''))
            cf_prompt = example.get('counterfactual', example.get('patch_prefix', ''))
            if not base_prompt or not cf_prompt:
                continue

            base_snippet = base_prompt[:100]

            # 全子空间软门控（含硬 ablation）
            if include_global:
                rows = ablate_feature_with_sfc(
                    model,
                    submodules_stash,
                    dictionaries,
                    base_prompt,
                    cf_prompt,
                    mediator,
                    feature_indices=None,
                    alpha_values=alpha_values,
                    edit_label='soft_gate_global',
                )
                meta = {
                    'mediator_layer': layer_idx,
                    'mediator_type': comp_type,
                    'mediator_head': mediator_head,
                    'mediator_nie': mediator.get('nie'),
                    'patching_effect_nie': patch_total_effect,
                    'feature_idx': -1,
                    'feature_source': 'global',
                    'feature_score': None,
                    'example_id': example_idx,
                    'example_base': base_snippet,
                    'mediator_category': mediator_category,
                }
                record_rows(rows, meta)

            # 目标特征的软门控/离散 ablation
            if include_feature_gating and topk_features:
                for feat_idx in topk_features:
                    rows = ablate_feature_with_sfc(
                        model,
                        submodules_stash,
                        dictionaries,
                        base_prompt,
                        cf_prompt,
                        mediator,
                        feature_indices=[int(feat_idx)],
                        alpha_values=alpha_values,
                        edit_label='soft_gate_feature',
                    )
                    meta = {
                        'mediator_layer': layer_idx,
                        'mediator_type': comp_type,
                        'mediator_head': mediator_head,
                        'mediator_nie': mediator.get('nie'),
                        'patching_effect_nie': patch_total_effect,
                        'feature_idx': int(feat_idx),
                        'feature_source': 'topk',
                        'feature_score': float(feature_scores[feat_idx]) if len(feature_scores) > feat_idx else None,
                        'example_id': example_idx,
                        'example_base': base_snippet,
                        'mediator_category': mediator_category,
                    }
                    record_rows(rows, meta)

            # 随机基线
            if include_random and random_features:
                for feat_idx in random_features:
                    rows = ablate_feature_with_sfc(
                        model,
                        submodules_stash,
                        dictionaries,
                        base_prompt,
                        cf_prompt,
                        mediator,
                        feature_indices=[int(feat_idx)],
                        alpha_values=alpha_values,
                        edit_label='random_soft_gate',
                    )
                    meta = {
                        'mediator_layer': layer_idx,
                        'mediator_type': comp_type,
                        'mediator_head': mediator_head,
                        'mediator_nie': mediator.get('nie'),
                        'patching_effect_nie': patch_total_effect,
                        'feature_idx': int(feat_idx),
                        'feature_source': 'random',
                        'feature_score': float(feature_scores[feat_idx]) if len(feature_scores) > feat_idx else None,
                        'example_id': example_idx,
                        'example_base': base_snippet,
                        'mediator_category': mediator_category,
                    }
                    record_rows(rows, meta)

            # head-off 基线
            if include_head_off:
                rows = ablate_feature_with_sfc(
                    model,
                    submodules_stash,
                    dictionaries,
                    base_prompt,
                    cf_prompt,
                    mediator,
                    feature_indices=None,
                    alpha_values=[1.0, 0.0],
                    edit_label='head_off',
                )
                meta = {
                    'mediator_layer': layer_idx,
                    'mediator_type': comp_type,
                    'mediator_head': mediator_head,
                    'mediator_nie': mediator.get('nie'),
                    'patching_effect_nie': patch_total_effect,
                    'feature_idx': -1,
                    'feature_source': 'head_off',
                    'feature_score': None,
                    'example_id': example_idx,
                    'example_base': base_snippet,
                    'mediator_category': mediator_category,
                }
                record_rows(rows, meta)

        # 聚合 dataset-level 统计
        for buffer in aggregate_buffer.values():
            rows = buffer['rows']
            if not rows:
                continue
            meta = buffer['meta'].copy()
            meta.update({
                'example_id': 'aggregate',
                'example_base': 'AGGREGATE',
                'example_count': len(rows),
                'analysis': 'aggregate',
            })
            meta['nie_original'] = float(np.mean([r['nie_original'] for r in rows]))
            meta['nie_ablated'] = float(np.mean([r['nie_ablated'] for r in rows]))
            meta['delta_nie'] = float(np.mean([r['delta_nie'] for r in rows]))
            meta['ppl_original'] = float(np.mean([r['ppl_original'] for r in rows]))
            meta['ppl_ablated'] = float(np.mean([r['ppl_ablated'] for r in rows]))
            meta['delta_ppl'] = float(np.mean([r['delta_ppl'] for r in rows]))
            if abs(meta['nie_original']) > 1e-9:
                meta['remaining_ratio'] = float(meta['nie_ablated'] / meta['nie_original'])
            else:
                meta['remaining_ratio'] = float('nan')
            all_results.append(meta)

    print(f"\n  ✓ 完成！共收集 {len(all_results)} 条记录（含聚合统计）")

    # 6. 保存结果并绘图
    print("\n[6/6] 保存结果...")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f'nie_vs_ppl_{timestamp}.csv')

    if all_results:
        aggregate_rows = [r for r in all_results if str(r.get('example_id')) == 'aggregate']
        summary_rows = []

        if aggregate_rows and nie_thresholds:
            grouped: Dict[tuple, List[Dict]] = defaultdict(list)
            for row in aggregate_rows:
                key = (
                    row.get('mediator_layer'),
                    row.get('mediator_head'),
                    row.get('mediator_type'),
                    row.get('mediator_category'),
                    row.get('edit_label'),
                    row.get('feature_source'),
                )
                grouped[key].append(row)

            for threshold in nie_thresholds:
                for key, rows in grouped.items():
                    candidates = [r for r in rows if not math.isnan(r.get('remaining_ratio', float('nan'))) and r['remaining_ratio'] <= threshold]
                    if not candidates:
                        continue
                    best = min(candidates, key=lambda r: (abs(r['delta_ppl']), r['alpha']))
                    summary = best.copy()
                    summary.update({
                        'analysis': 'matched_threshold',
                        'threshold': float(threshold),
                    })
                    summary_rows.append(summary)

        if aggregate_rows:
            grouped_for_pareto: Dict[tuple, List[Dict]] = defaultdict(list)
            for row in aggregate_rows:
                key = (
                    row.get('mediator_category'),
                    row.get('edit_label'),
                    row.get('feature_source'),
                )
                grouped_for_pareto[key].append(row)

            for key, rows in grouped_for_pareto.items():
                sorted_rows = sorted(rows, key=lambda r: (abs(r['delta_ppl']), r['alpha']))
                best_so_far = float('inf')
                rank = 0
                for row in sorted_rows:
                    abs_nie = abs(row['delta_nie'])
                    if abs_nie < best_so_far - 1e-9:
                        best_so_far = abs_nie
                        summary = row.copy()
                        summary.update({
                            'analysis': 'pareto_frontier',
                            'pareto_rank': rank,
                        })
                        summary_rows.append(summary)
                        rank += 1

        if summary_rows:
            all_results.extend(summary_rows)

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = sorted({key for row in all_results for key in row.keys()})
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"  ✓ CSV 已保存: {csv_path}")
        plot_nie_vs_ppl(all_results, output_dir, timestamp)

    print("\n" + "=" * 70)
    print("实验完成！")
    print("=" * 70)


def plot_nie_vs_ppl(results: List[Dict], output_dir: str, timestamp: str):
    """绘制 NIE vs ΔPPL 散点图"""
    plt.figure(figsize=(10, 6))

    aggregate_rows = [r for r in results if str(r.get('example_id')) == 'aggregate']
    plot_rows = aggregate_rows if aggregate_rows else results

    abs_nie = [abs(r['delta_nie']) for r in plot_rows]
    delta_ppl = [r['delta_ppl'] for r in plot_rows]
    layers = [r['mediator_layer'] for r in plot_rows]
    
    scatter = plt.scatter(abs_nie, delta_ppl, c=layers, cmap='viridis', alpha=0.6, s=50)
    plt.colorbar(scatter, label='Layer')
    
    plt.xlabel('|ΔNIE| (Absolute Indirect Effect)', fontsize=12)
    plt.ylabel('ΔPPL (Perplexity Change)', fontsize=12)
    plt.title('Bias-Perplexity Pareto: Feature Ablation (SFC)', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plots_dir = os.path.join(output_dir, '..', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, f'nie_vs_ppl_{timestamp}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ 图表已保存: {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="NIE - Feature Ablation 实验（使用完整 SFC 源码）"
    )
    
    parser.add_argument('--model', type=str, default='gpt2')
    parser.add_argument('--topk_json', type=str, required=True)
    parser.add_argument('--eval_data', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--num_features', type=int, default=10)
    parser.add_argument('--alphas', type=str, default='1.0,0.75,0.5,0.25,0.0',
                        help='逗号分隔的软门控 alpha 值，默认 "1.0,0.75,0.5,0.25,0.0"')
    parser.add_argument('--max_examples', type=int, default=3,
                        help='每个 mediator 评估的最大样本数，<=0 表示使用全部样本')
    parser.add_argument('--gating_target', type=str, choices=['global', 'feature', 'both'], default='both',
                        help='选择仅对整个子空间、特征子空间或二者都进行门控')
    parser.add_argument('--random_baseline', action='store_true',
                        help='启用随机特征门控作为基线')
    parser.add_argument('--include_head_off', action='store_true',
                        help='启用 head-off 基线')
    parser.add_argument('--control_mediators', type=int, default=0,
                        help='额外选择若干低 NIE mediator 作为控制组')
    parser.add_argument('--nie_thresholds', type=str, default='0.75,0.5',
                        help='逗号分隔的剩余 NIE 比例阈值，例如 "0.75,0.5"')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--device', type=str, 
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()

    def parse_alpha_list(alpha_str: str) -> List[float]:
        values: List[float] = []
        for part in alpha_str.split(','):
            part = part.strip()
            if not part:
                continue
            try:
                values.append(float(part))
            except ValueError:
                raise ValueError(f"无法解析 alpha 值: {part}")
        return values

    alpha_values = parse_alpha_list(args.alphas)
    include_global = args.gating_target in ('global', 'both')
    include_feature_gating = args.gating_target in ('feature', 'both')

    def parse_threshold_list(threshold_str: str) -> List[float]:
        values: List[float] = []
        for part in threshold_str.split(','):
            part = part.strip()
            if not part:
                continue
            try:
                values.append(float(part))
            except ValueError:
                raise ValueError(f"无法解析 NIE 阈值: {part}")
        return values

    nie_thresholds = parse_threshold_list(args.nie_thresholds) if args.nie_thresholds else []

    run_experiment(
        model_name=args.model,
        topk_mediators_path=args.topk_json,
        eval_data_path=args.eval_data,
        output_dir=args.output_dir,
        topk=args.topk,
        num_features_per_mediator=args.num_features,
        alpha_values=alpha_values,
        max_examples=args.max_examples,
        include_random=args.random_baseline,
        include_global=include_global,
        include_feature_gating=include_feature_gating,
        include_head_off=args.include_head_off,
        control_mediators=args.control_mediators,
        nie_thresholds=nie_thresholds,
        seed=args.seed,
        device=args.device,
    )


if __name__ == '__main__':
    main()
