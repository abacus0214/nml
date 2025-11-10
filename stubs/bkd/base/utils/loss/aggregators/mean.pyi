from bkd.base.utils.loss.aggregators.base import LossAggregatorABC
from torch import Tensor
from typing import Iterable

class NPMeanAggregator[LossT](LossAggregatorABC[LossT]):
    def aggregate(self, losses: Iterable[LossT]) -> LossT: ...

class TorchMeanAggregator[LossT: Tensor](LossAggregatorABC[LossT]):
    def aggregate(self, losses: Iterable[LossT]) -> LossT: ...
