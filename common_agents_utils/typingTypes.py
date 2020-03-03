import torch
import numpy as np
from typing import Union, Dict, Tuple, Any, Iterable, Optional, Callable


TT = torch.Tensor
NpA = np.ndarray
npTT = Union[TT, NpA]
StatType = Dict[str, Any]
TTStat = Tuple[TT, StatType]
NpAStat = Tuple[NpA, StatType]
NpAOrNpAStat = Union[NpA, NpAStat]
TTOrTTStat = Union[TT, TTStat]
