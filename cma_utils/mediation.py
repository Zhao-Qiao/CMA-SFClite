"""Causal Mediation Analysis 核心功能

提供 NIE 计算、mediator 操作等功能
基于 Vig et al. 2020 的方法
"""

from typing import List, Dict, Tuple
import torch
import json
import csv


def get_token_id(tokenizer, text: str) -> int:
    """获取文本的首个 token id"""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError(f"无法将 '{text}' tokenize 为有效 token")
    return ids[0]


def compute_nie_for_head(
    model,
    base_prompt: str,
    cf_prompt: str,
    layer_idx: int,
    head_idx: int,
    metric_fn=None,  # 可选的自定义度量函数
) -> float:
    """
    计算单个注意力头的 NIE（Natural Indirect Effect）
    
    参数：
    - model: nnsight.LanguageModel
    - base_prompt: base 句子
    - cf_prompt: counterfactual 句子
    - layer_idx: 层索引
    - head_idx: 头索引
    - metric_fn: 可选的自定义度量函数 (model) -> scalar
    
    返回：NIE 值
    
    算法（Interchange Intervention）：
    1. counterfactual 轨迹，保存该头的中介激活 Z_cf
    2. base 轨迹（未干预）→ metric_clean
    3. base 轨迹（干预：Z → Z_cf）→ metric_int
    4. NIE = metric_int - metric_clean
    """
    attn_dim = model.config.n_embd // model.config.n_head
    h_start = head_idx * attn_dim
    h_end = (head_idx + 1) * attn_dim
    
    with torch.no_grad():
        # 步骤 1：counterfactual 轨迹，保存该头的中介激活
        with model.trace(cf_prompt):
            # 对于 GPT-2 类模型
            if hasattr(model, 'transformer'):
                z_cf = model.transformer.h[layer_idx].attn.c_proj.input[0, -1, h_start:h_end].save()
            # 对于 Pythia 类模型
            elif hasattr(model, 'gpt_neox'):
                z_cf = model.gpt_neox.layers[layer_idx].attention.dense.input[0, -1, h_start:h_end].save()
            else:
                raise ValueError(f"Unsupported model type: {type(model)}")
        z_cf_val = z_cf.detach()
        
        # 步骤 2：base 轨迹（未干预）
        with model.trace(base_prompt):
            if metric_fn is not None:
                metric_clean = metric_fn(model).save()
            else:
                logits_clean = model.output.logits[0, -1, :].save()
        
        # 步骤 3：base 轨迹（干预：替换该头为 cf 的激活）
        with model.trace(base_prompt):
            if hasattr(model, 'transformer'):
                model.transformer.h[layer_idx].attn.c_proj.input[0, -1, h_start:h_end] = z_cf_val
            elif hasattr(model, 'gpt_neox'):
                model.gpt_neox.layers[layer_idx].attention.dense.input[0, -1, h_start:h_end] = z_cf_val
            
            if metric_fn is not None:
                metric_int = metric_fn(model).save()
            else:
                logits_int = model.output.logits[0, -1, :].save()
        
        # 计算 NIE
        if metric_fn is not None:
            nie = (metric_int - metric_clean).item()
        else:
            # 默认：使用 logit 差异
            nie = (logits_int - logits_clean).mean().item()
    
    return nie


def compute_nie_for_mediator(
    model,
    base_prompt: str,
    cf_prompt: str,
    mediator_type: str,  # 'attn' or 'mlp'
    layer_idx: int,
    component_idx: int = None,  # 对 attn 是 head_idx，对 mlp 可选
    metric_fn=None,
) -> float:
    """
    计算任意 mediator 的 NIE
    
    支持：
    - 注意力头 (mediator_type='attn', 需要 component_idx)
    - MLP 层 (mediator_type='mlp')
    """
    if mediator_type == 'attn':
        assert component_idx is not None, "attn mediator 需要指定 component_idx (head_idx)"
        return compute_nie_for_head(model, base_prompt, cf_prompt, layer_idx, component_idx, metric_fn)
    
    elif mediator_type == 'mlp':
        # MLP NIE 计算
        with torch.no_grad():
            # 步骤 1：保存 counterfactual 的 MLP 输出
            with model.trace(cf_prompt):
                if hasattr(model, 'transformer'):
                    z_cf = model.transformer.h[layer_idx].mlp.output[0, -1, :].save()
                elif hasattr(model, 'gpt_neox'):
                    z_cf = model.gpt_neox.layers[layer_idx].mlp.output[0, -1, :].save()
            z_cf_val = z_cf.detach()
            
            # 步骤 2：base clean
            with model.trace(base_prompt):
                if metric_fn is not None:
                    metric_clean = metric_fn(model).save()
                else:
                    logits_clean = model.output.logits[0, -1, :].save()
            
            # 步骤 3：base intervened
            with model.trace(base_prompt):
                if hasattr(model, 'transformer'):
                    model.transformer.h[layer_idx].mlp.output[0, -1, :] = z_cf_val
                elif hasattr(model, 'gpt_neox'):
                    model.gpt_neox.layers[layer_idx].mlp.output[0, -1, :] = z_cf_val
                
                if metric_fn is not None:
                    metric_int = metric_fn(model).save()
                else:
                    logits_int = model.output.logits[0, -1, :].save()
            
            # 计算 NIE
            if metric_fn is not None:
                nie = (metric_int - metric_clean).item()
            else:
                nie = (logits_int - logits_clean).mean().item()
        
        return nie
    
    else:
        raise ValueError(f"Unsupported mediator_type: {mediator_type}")


def load_topk_mediators(csv_path: str, topk: int = 10) -> List[Dict]:
    """
    从 CMA 输出的 CSV 文件加载 Top-K mediators
    
    返回：[{
        'layer': int,
        'head': int,
        'nie': float,
        'abs_nie': float,
        'rank': int
    }, ...]
    """
    mediators = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['rank']:  # 跳过注释行
                try:
                    mediators.append({
                        'rank': int(row['rank']),
                        'layer': int(row['layer']),
                        'head': int(row['head']),
                        'nie': float(row['nie']),
                        'abs_nie': float(row['abs_nie']),
                    })
                except (ValueError, KeyError):
                    continue
    
    # 按 abs_nie 排序并取 Top-K
    mediators = sorted(mediators, key=lambda x: x['abs_nie'], reverse=True)[:topk]
    return mediators


def load_topk_mediators_from_json(json_path: str, topk: int = 10) -> List[Dict]:
    """
    从 JSON 文件加载 Top-K mediators
    
    格式：{
        "mediators": [
            {"type": "attn", "layer": 0, "head": 5, "nie": 0.123},
            ...
        ]
    }
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    mediators = data.get('mediators', [])
    # 添加 abs_nie 字段
    for m in mediators:
        m['abs_nie'] = abs(m['nie'])
    
    # 排序并取 Top-K
    mediators = sorted(mediators, key=lambda x: x['abs_nie'], reverse=True)[:topk]
    return mediators

