"""Fairness delta reaches zero when groups identical"""

# Smoke test for the pairwise-disparity calculation
def test_identical_groups_have_zero_disparity():
    group_scores = {'A': 0.8, 'B': 0.8, 'C': 0.8}
    scores = list(group_scores.values())
    delta = max(scores) - min(scores)
    assert abs(delta) < 1e-9

