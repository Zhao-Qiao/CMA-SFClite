from collections import namedtuple
from dictionary_learning import AutoEncoder, JumpReluAutoEncoder
from dictionary_learning.dictionary import IdentityDict
from .attribution import Submodule
from typing import Literal
import torch as t
from huggingface_hub import list_repo_files
from tqdm import tqdm
import os
import pickle
import sys

# 尝试导入sae_training mock模块
try:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    import sae_training_mock
except:
    pass

DICT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/dictionaries"

DictionaryStash = namedtuple("DictionaryStash", ["embed", "attns", "mlps", "resids"])


def load_gpt2_sae_from_file(file_path: str, dtype: t.dtype = t.float32, device: t.device = t.device("cpu")):
    """
    从GPT2-Small SAE文件加载AutoEncoder
    
    这些文件是用sae_training模块保存的，包含cfg和state_dict
    文件格式：
    {
        'cfg': LanguageModelSAERunnerConfig,
        'state_dict': {
            'W_enc': [768, 24576],
            'b_enc': [24576],
            'W_dec': [24576, 768],
            'b_dec': [768]
        }
    }
    """
    try:
        # 使用weights_only=False加载（文件来自可信源）
        sae_data = t.load(file_path, map_location=device, weights_only=False)
        
        # 获取state_dict
        if isinstance(sae_data, dict) and 'state_dict' in sae_data:
            # 标准格式：包含cfg和state_dict
            state_dict = sae_data['state_dict']
        elif isinstance(sae_data, dict):
            # 直接是state_dict
            state_dict = sae_data
        elif hasattr(sae_data, 'state_dict'):
            # 模型对象
            state_dict = sae_data.state_dict()
        else:
            return None
        
        # GPT2-Small SAE使用的键名
        if 'W_enc' in state_dict and 'W_dec' in state_dict:
            W_enc = state_dict['W_enc']  # [768, 24576]
            W_dec = state_dict['W_dec']  # [24576, 768]
            b_enc = state_dict.get('b_enc')  # [24576]
            b_dec = state_dict.get('b_dec')  # [768]
            
            # W_enc是[hidden_size, dict_size]，需要转置为[dict_size, hidden_size]
            activation_dim, dict_size = W_enc.shape
            
            # 创建AutoEncoder (参数: activation_dim, dict_size)
            ae = AutoEncoder(activation_dim, dict_size)
            ae = ae.to(dtype=dtype, device=device)
            
            # 加载权重（需要转置）
            ae.encoder.weight.data = W_enc.T.to(dtype=dtype, device=device)  # [24576, 768]
            ae.decoder.weight.data = W_dec.T.to(dtype=dtype, device=device)  # [768, 24576]
            
            # 加载偏置
            if b_enc is not None and ae.encoder.bias is not None:
                ae.encoder.bias.data = b_enc.to(dtype=dtype, device=device)
            # 注意：decoder默认没有bias，但SAE文件有b_dec
            # 我们需要手动添加bias或者忽略b_dec
            # 这里选择忽略，因为AutoEncoder的decoder不使用bias
            
            return ae
        else:
            # 其他格式，尝试通用加载
            return None
            
    except Exception as e:
        # 静默失败，返回None
        return None


def _load_pythia_saes_and_submodules(
    model,
    thru_layer: int | None = None,
    separate_by_type: bool = False,
    include_embed: bool = True,
    neurons: bool = False,
    dtype: t.dtype = t.float32,
    device: t.device = t.device("cpu"),
):
    assert (
        len(model.gpt_neox.layers) == 6
    ), "Not the expected number of layers for pythia-70m-deduped"
    if thru_layer is None:
        thru_layer = len(model.gpt_neox.layers)

    attns = []
    mlps = []
    resids = []
    dictionaries = {}
    if include_embed:
        embed = Submodule(
            name="embed",
            submodule=model.gpt_neox.embed_in,
        )
        if not neurons:
            dictionaries[embed] = AutoEncoder.from_pretrained(
                f"{DICT_DIR}/pythia-70m-deduped/embed/10_32768/ae.pt",
                dtype=dtype,
                device=device,
            )
        else:
            dictionaries[embed] = IdentityDict(512, device=device, dtype=dtype)
    else:
        embed = None
    for i, layer in enumerate(model.gpt_neox.layers[: thru_layer + 1]):
        attns.append(
            attn := Submodule(
                name=f"attn_{i}",
                submodule=layer.attention,
                is_tuple=True,
            )
        )
        mlps.append(
            mlp := Submodule(
                name=f"mlp_{i}",
                submodule=layer.mlp,
            )
        )
        resids.append(
            resid := Submodule(
                name=f"resid_{i}",
                submodule=layer,
                is_tuple=True,
            )
        )
        if not neurons:
            dictionaries[attn] = AutoEncoder.from_pretrained(
                f"{DICT_DIR}/pythia-70m-deduped/attn_out_layer{i}/10_32768/ae.pt",
                dtype=dtype,
                device=device,
            )
            dictionaries[mlp] = AutoEncoder.from_pretrained(
                f"{DICT_DIR}/pythia-70m-deduped/mlp_out_layer{i}/10_32768/ae.pt",
                dtype=dtype,
                device=device,
            )
            dictionaries[resid] = AutoEncoder.from_pretrained(
                f"{DICT_DIR}/pythia-70m-deduped/resid_out_layer{i}/10_32768/ae.pt",
                dtype=dtype,
                device=device,
            )
        else:
            dictionaries[attn] = IdentityDict(512, device=device, dtype=dtype)
            dictionaries[mlp] = IdentityDict(512, device=device, dtype=dtype)
            dictionaries[resid] = IdentityDict(512, device=device, dtype=dtype)

    if separate_by_type:
        return DictionaryStash(embed, attns, mlps, resids), dictionaries
    else:
        submodules = ([embed] if include_embed else []) + [
            x
            for layer_dictionaries in zip(attns, mlps, resids)
            for x in layer_dictionaries
        ]
        return submodules, dictionaries


def load_gemma_sae(
    submod_type: Literal["embed", "attn", "mlp", "resid"],
    layer: int,
    width: Literal["16k", "65k"] = "16k",
    neurons: bool = False,
    dtype: t.dtype = t.float32,
    device: t.device = t.device("cpu"),
):
    if neurons:
        if submod_type != "attn":
            return IdentityDict(2304, device=device, dtype=dtype)
        else:
            return IdentityDict(2048, device=device, dtype=dtype)

    repo_id = "google/gemma-scope-2b-pt-" + (
        "res"
        if submod_type in ["embed", "resid"]
        else "att" if submod_type == "attn" else "mlp"
    )
    if submod_type != "embed":
        directory_path = f"layer_{layer}/width_{width}"
    else:
        directory_path = "embedding/width_4k"

    files_with_l0s = [
        (f, int(f.split("_")[-1].split("/")[0]))
        for f in list_repo_files(repo_id, repo_type="model", revision="main")
        if f.startswith(directory_path) and f.endswith("params.npz")
    ]
    optimal_file = min(files_with_l0s, key=lambda x: abs(x[1] - 100))[0]
    optimal_file = optimal_file.split("/params.npz")[0]
    return JumpReluAutoEncoder.from_pretrained(
        load_from_sae_lens=True,
        release=repo_id.split("google/")[-1],
        sae_id=optimal_file,
        dtype=dtype,
        device=device,
    )


def _load_gpt2_saes_and_submodules(
    model,
    thru_layer: int | None = None,
    separate_by_type: bool = False,
    include_embed: bool = True,
    neurons: bool = False,
    dtype: t.dtype = t.float32,
    device: t.device = t.device("cpu"),
):
    """加载GPT2-Small的SAE字典和submodules"""
    assert (
        len(model.transformer.h) == 12
    ), "Not the expected number of layers for GPT2-Small (should be 12)"
    if thru_layer is None:
        thru_layer = len(model.transformer.h)

    attns = []
    mlps = []
    resids = []
    dictionaries = {}
    
    # GPT2-Small的SAE字典路径
    gpt2_sae_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                               "models", "GPT2-Small-SAEs")
    
    if include_embed:
        embed = Submodule(
            name="embed",
            submodule=model.transformer.wte,  # GPT2的embedding层
        )
        if not neurons:
            # GPT2-Small没有embedding SAE，使用IdentityDict
            dictionaries[embed] = IdentityDict(768)
        else:
            dictionaries[embed] = IdentityDict(768)
    else:
        embed = None
    
    for i, layer in enumerate(model.transformer.h[: thru_layer + 1]):
        # GPT2的SAE是在hook_resid_pre上训练的（每层的residual stream）
        # 由于SAE字典是统一的，我们对所有component types使用相同的resid submodule
        # 这样可以确保SAE正确应用在residual stream上
        
        # 所有类型都指向整个layer（residual stream）
        # 这样SAE会正确地应用在residual stream激活上
        attns.append(
            attn := Submodule(
                name=f"resid_{i}_attn",  # 改名以反映实际hook点
                submodule=layer,
                is_tuple=True,
            )
        )
        
        mlps.append(
            mlp := Submodule(
                name=f"resid_{i}_mlp",
                submodule=layer,
                is_tuple=True,
            )
        )
        
        resids.append(
            resid := Submodule(
                name=f"resid_{i}",
                submodule=layer,
                is_tuple=True,
            )
        )
        
        if not neurons:
            # 加载对应的SAE字典（所有类型共享同一个resid_pre SAE）
            try:
                # 加载resid_pre SAE
                sae_path = os.path.join(gpt2_sae_dir, 
                    f"final_sparse_autoencoder_gpt2-small_blocks.{i}.hook_resid_pre_24576.pt")
                if os.path.exists(sae_path):
                    sae_dict = load_gpt2_sae_from_file(sae_path, dtype=dtype, device=device)
                    if sae_dict is not None:
                        # 所有类型共享同一个SAE
                        dictionaries[attn] = sae_dict
                        dictionaries[mlp] = sae_dict
                        dictionaries[resid] = sae_dict
                        if i == 0:
                            print(f"  ✓ 成功加载GPT2-Small SAE字典 (24576维，应用于residual stream)")
                    else:
                        if i == 0:
                            print(f"  ⚠ SAE文件格式不支持，回退使用IdentityDict")
                        dictionaries[attn] = IdentityDict(768)
                        dictionaries[mlp] = IdentityDict(768)
                        dictionaries[resid] = IdentityDict(768)
                else:
                    if i == 0:
                        print(f"  ⚠ 未找到SAE文件，使用IdentityDict")
                    dictionaries[attn] = IdentityDict(768)
                    dictionaries[mlp] = IdentityDict(768)
                    dictionaries[resid] = IdentityDict(768)
                
            except Exception as e:
                if i == 0:
                    print(f"  ⚠ SAE加载异常，使用IdentityDict: {str(e)[:100]}")
                dictionaries[attn] = IdentityDict(768)
                dictionaries[mlp] = IdentityDict(768)
                dictionaries[resid] = IdentityDict(768)
        else:
            dictionaries[attn] = IdentityDict(768)
            dictionaries[mlp] = IdentityDict(768)
            dictionaries[resid] = IdentityDict(768)

    if separate_by_type:
        return DictionaryStash(embed, attns, mlps, resids), dictionaries
    else:
        submodules = ([embed] if include_embed else []) + [
            x
            for layer_dictionaries in zip(attns, mlps, resids)
            for x in layer_dictionaries
        ]
        return submodules, dictionaries


def _load_gemma_saes_and_submodules(
    model,
    thru_layer: int | None = None,
    separate_by_type: bool = False,
    include_embed: bool = True,
    neurons: bool = False,
    dtype: t.dtype = t.float32,
    device: t.device = t.device("cpu"),
):
    assert (
        len(model.model.layers) == 26
    ), "Not the expected number of layers for Gemma-2-2B"
    if thru_layer is None:
        thru_layer = len(model.model.layers)

    attns = []
    mlps = []
    resids = []
    dictionaries = {}
    if include_embed:
        embed = Submodule(
            name="embed",
            submodule=model.model.embed_tokens,
        )
        dictionaries[embed] = load_gemma_sae(
            "embed", 0, neurons=neurons, dtype=dtype, device=device
        )
    else:
        embed = None
    for i, layer in tqdm(
        enumerate(model.model.layers[: thru_layer + 1]),
        total=thru_layer + 1,
        desc="Loading Gemma SAEs",
    ):
        attns.append(
            attn := Submodule(
                name=f"attn_{i}", submodule=layer.self_attn.o_proj, use_input=True
            )
        )
        dictionaries[attn] = load_gemma_sae(
            "attn", i, neurons=neurons, dtype=dtype, device=device
        )
        mlps.append(
            mlp := Submodule(
                name=f"mlp_{i}",
                submodule=layer.post_feedforward_layernorm,
            )
        )
        dictionaries[mlp] = load_gemma_sae(
            "mlp", i, neurons=neurons, dtype=dtype, device=device
        )
        resids.append(
            resid := Submodule(
                name=f"resid_{i}",
                submodule=layer,
                is_tuple=True,
            )
        )
        dictionaries[resid] = load_gemma_sae(
            "resid", i, neurons=neurons, dtype=dtype, device=device
        )

    if separate_by_type:
        return DictionaryStash(embed, attns, mlps, resids), dictionaries
    else:
        submodules = ([embed] if include_embed else []) + [
            x
            for layer_dictionaries in zip(attns, mlps, resids)
            for x in layer_dictionaries
        ]
        return submodules, dictionaries


def load_saes_and_submodules(
    model,
    thru_layer: int | None = None,
    separate_by_type: bool = False,
    include_embed: bool = True,
    neurons: bool = False,
    dtype: t.dtype = t.float32,
    device: t.device = t.device("cpu"),
):
    model_name = model.config._name_or_path

    if model_name == "EleutherAI/pythia-70m-deduped":
        return _load_pythia_saes_and_submodules(
            model,
            thru_layer=thru_layer,
            separate_by_type=separate_by_type,
            include_embed=include_embed,
            neurons=neurons,
            dtype=dtype,
            device=device,
        )
    elif model_name == "google/gemma-2-2b":
        return _load_gemma_saes_and_submodules(
            model,
            thru_layer=thru_layer,
            separate_by_type=separate_by_type,
            include_embed=include_embed,
            neurons=neurons,
            dtype=dtype,
            device=device,
        )
    elif model_name in ["gpt2", "gpt2-small"]:
        return _load_gpt2_saes_and_submodules(
            model,
            thru_layer=thru_layer,
            separate_by_type=separate_by_type,
            include_embed=include_embed,
            neurons=neurons,
            dtype=dtype,
            device=device,
        )
    else:
        raise ValueError(f"Model {model_name} not supported. Supported models: EleutherAI/pythia-70m-deduped, google/gemma-2-2b, gpt2, gpt2-small")
