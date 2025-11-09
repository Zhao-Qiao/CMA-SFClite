"""SFC (Sparse Feature Circuits) utilities - 直接使用原始源码"""

from .activation_utils import SparseAct
from .dictionary_loading_utils import load_saes_and_submodules, DictionaryStash
from .attribution import patching_effect, jvp, EffectOut
from .ablation import run_with_ablations
from .circuit import get_circuit

__all__ = [
    'SparseAct',
    'load_saes_and_submodules',
    'DictionaryStash',
    'patching_effect',
    'jvp',
    'EffectOut',
    'run_with_ablations',
    'get_circuit',
]
