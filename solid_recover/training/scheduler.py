r"""Custom 4-phase learning rate scheduler: warmup -> steady -> cosine -> min lr.

Direct port of ``lr_scheduler.sr_scheduler`` with PascalCase class name.

Schedule shape (ASCII)::

    lr
    ^
    |               ________________
    | base_lr    . /                \
    |          ./                    \.
    |        ./                        \.
    |      ./                            \._______________
    |    ./                                   min_lr
    |  ./
    | /
    |/
    +---|--------------|--------------------|-----------------> step
        0            w_end        w_end + s_end       w_end + s_end + c_end

    w_end = warmup_steps
    s_end = steady_1_steps              (plateau held at base_lr)
    c_end = cosine_anneal_steps          (cosine decay to min_lr)

Phases
------
1. **Linear warmup** for ``warmup_steps``: LR grows linearly from 0 to
   ``base_lr`` (i.e. the per-group optimizer LR).
2. **Plateau** for ``steady_1_steps``: LR held constant at ``base_lr``.
3. **Cosine anneal** over ``cosine_anneal_steps``: LR follows a half-cosine
   curve from ``base_lr`` down to ``min_lr``.
4. **Floor**: LR held constant at ``min_lr`` for any further step.

Set ``warmup_steps=0`` or ``steady_1_steps=0`` to skip the corresponding
phase; ``cosine_anneal_steps=0`` collapses directly to ``min_lr`` right after
the plateau.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Union

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class SRScheduler(_LRScheduler):
    """LR schedule: linear warmup -> constant -> cosine anneal -> constant min.

    Parameters
    ----------
    optimizer:
        Wrapped optimizer.
    warmup_steps:
        Number of linear-warmup steps (can be 0).
    steady_1_steps:
        Number of steps held at ``base_lr`` after warmup (can be 0).
    cosine_anneal_steps:
        Number of steps over which the LR cosine-anneals down to ``min_lr``.
    min_lr:
        Either a single float or a per-param-group list.
    last_epoch:
        Forwarded to ``_LRScheduler``.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        steady_1_steps: int,
        cosine_anneal_steps: int,
        min_lr: Union[float, Sequence[float]],
        last_epoch: int = -1,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.steady_1_steps = steady_1_steps
        self.cosine_anneal_steps = cosine_anneal_steps

        if isinstance(min_lr, (int, float)):
            self.min_lrs: List[float] = [float(min_lr)] * len(optimizer.param_groups)
        else:
            self.min_lrs = [float(v) for v in min_lr]

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:  # type: ignore[override]
        base_lrs = self.base_lrs
        step = self.last_epoch

        # Phase 1: linear warmup
        if step < self.warmup_steps:
            return [base * (step / max(1, self.warmup_steps)) for base in base_lrs]

        # Phase 2: constant at base_lr
        if step < self.warmup_steps + self.steady_1_steps:
            return list(base_lrs)

        # Phase 3: cosine annealing to min_lr
        cosine_end = self.warmup_steps + self.steady_1_steps + self.cosine_anneal_steps
        if step < cosine_end:
            step_in_cosine = step - (self.warmup_steps + self.steady_1_steps)
            return [
                self.min_lrs[i]
                + 0.5
                * (base - self.min_lrs[i])
                * (1 + math.cos(math.pi * step_in_cosine / max(1, self.cosine_anneal_steps)))
                for i, base in enumerate(base_lrs)
            ]

        # Phase 4: constant at min_lr
        return list(self.min_lrs)
