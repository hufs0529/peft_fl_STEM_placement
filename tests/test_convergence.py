"""계획서 v2 §5.3 수렴 기준(3라운드 연속 1% 미만 개선) 단위 테스트."""

from src.convergence import ConvergenceTracker


def test_converges_when_plateaued():
    tracker = ConvergenceTracker(min_improvement_pct=1.0, patience_rounds=3)
    # 2.0 -> 1.99(0.5% 개선) -> 1.98(0.5%) -> 1.97(0.5%): 전부 1% 미만 개선
    sequence = [2.0, 1.99, 1.98, 1.97]
    results = [tracker.update(v) for v in sequence]
    assert results[-1] is True, "3라운드 연속 1% 미만 개선인데 수렴으로 판정되지 않음"


def test_does_not_converge_with_large_improvement():
    tracker = ConvergenceTracker(min_improvement_pct=1.0, patience_rounds=3)
    # 큰 폭으로 계속 개선되는 시퀀스는 수렴 판정이 나면 안 됨
    sequence = [2.0, 1.5, 1.0, 0.5]
    results = [tracker.update(v) for v in sequence]
    assert results[-1] is False


def test_insufficient_history_never_converges():
    tracker = ConvergenceTracker(min_improvement_pct=1.0, patience_rounds=3)
    assert tracker.update(2.0) is False
    assert tracker.update(1.99) is False  # patience_rounds(3)에 아직 못 미침
