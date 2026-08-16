"""수렴 기준 실제 구현 (계획서 v2 §5.3): 3라운드 연속 1% 미만 개선이면 종료.

Actual implementation of the convergence criterion (plan v2 §5.3): stop if
improvement is below 1% for 3 consecutive rounds.
"""

from typing import List


class ConvergenceTracker:
    def __init__(self, min_improvement_pct: float = 1.0, patience_rounds: int = 3):
        self.min_improvement_pct = min_improvement_pct
        self.patience_rounds = patience_rounds
        self.history: List[float] = []

    def update(self, val_loss: float) -> bool:
        self.history.append(val_loss)
        if len(self.history) <= self.patience_rounds:
            return False

        recent_window = self.history[-(self.patience_rounds + 1):]
        improvements = []
        for i in range(1, len(recent_window)):
            prev, curr = recent_window[i - 1], recent_window[i]
            pct_improvement = 0.0 if prev == 0 else 100.0 * (prev - curr) / abs(prev)
            improvements.append(pct_improvement)

        return all(imp < self.min_improvement_pct for imp in improvements)

    def converged_at_round(self) -> int:
        return len(self.history)
