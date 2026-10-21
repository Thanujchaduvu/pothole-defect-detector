import pytest

from src.severity import coverage_sum, coverage_union, score, severity_from_coverage

W, H = 100, 100  # 10,000 px image


def test_no_detection():
    assert score([], W, H)["severity"] == "NONE"


@pytest.mark.parametrize("cov,label", [
    (0.5, "LOW"), (3.0, "LOW"), (3.1, "MEDIUM"), (10.0, "MEDIUM"),
    (10.1, "HIGH"), (20.0, "HIGH"), (20.1, "CRITICAL"), (80, "CRITICAL"),
])
def test_bands(cov, label):
    assert severity_from_coverage(cov) == label


def test_single_box_coverage():
    assert coverage_union([(0, 0, 10, 10)], W, H) == pytest.approx(1.0)
    assert coverage_sum([(0, 0, 10, 10)], W, H) == pytest.approx(1.0)


def test_overlap_union_vs_sum():
    boxes = [(0, 0, 50, 50), (0, 0, 50, 50)]  # identical boxes
    assert coverage_sum(boxes, W, H) == pytest.approx(50.0)
    assert coverage_union(boxes, W, H) == pytest.approx(25.0)


def test_boxes_clipped_to_image():
    assert coverage_union([(-20, -20, 200, 200)], W, H) == pytest.approx(100.0)


def test_score_critical():
    assert score([(0, 0, 60, 60)], W, H)["severity"] == "CRITICAL"
