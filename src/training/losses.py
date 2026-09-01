from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, Iterable, Sequence, Tuple

import torch


class FairnessRegulariser:
    """Demographic-parity style regulariser over protected attribute groups.

    Maintains an EMA of per-sample loss per group bucket. The penalty is the
    variance of group EMAs, pushing training toward equalised loss across groups.
    """

    def __init__(self, attributes: Sequence[str], ema_window: int = 512, weight: float = 0.1):
        self.attributes = tuple(attributes)
        self.weight = weight
        self.window = ema_window
        self._buckets: Dict[Tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=ema_window))

    def update_and_penalty(
        self,
        per_sample_loss: torch.Tensor,
        attributes: Sequence[Tuple[str, ...]],
    ) -> torch.Tensor:
        detached = per_sample_loss.detach().float().cpu().tolist()
        for loss_v, attr in zip(detached, attributes):
            for name, value in zip(self.attributes, attr):
                self._buckets[(name, value)].append(loss_v)

        group_means: Dict[str, list] = defaultdict(list)
        for (name, _), hist in self._buckets.items():
            if len(hist) >= 8:
                group_means[name].append(sum(hist) / len(hist))

        device = per_sample_loss.device
        penalties = []
        for name in self.attributes:
            means = group_means.get(name)
            if means and len(means) >= 2:
                t = torch.tensor(means, device=device)
                penalties.append(t.var(unbiased=False))
        if not penalties:
            return torch.zeros((), device=device)
        return self.weight * torch.stack(penalties).mean()

    def disparity(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for name in self.attributes:
            means = [sum(h) / len(h) for (n, _), h in self._buckets.items() if n == name and h]
            if len(means) >= 2:
                out[name] = max(means) - min(means)
            else:
                out[name] = 0.0
        return out


def kl_consistency_loss(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    import torch.nn.functional as F

    log_p = F.log_softmax(logits_a, dim=-1)
    q = F.softmax(logits_b, dim=-1)
    return F.kl_div(log_p, q, reduction="batchmean")
