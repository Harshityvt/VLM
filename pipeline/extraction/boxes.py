"""Box helpers shared by the detectors. All boxes are normalized (x1, y1, x2, y2)."""
from dataclasses import dataclass, field

Box = tuple[float, float, float, float]


@dataclass
class Det:
    label: str
    box: Box
    score: float
    extra: dict = field(default_factory=dict)


def parse_bounding_boxes(item: dict, min_score: dict[str, float] | float = 0.0) -> list[Det]:
    """Object-detection NIM response item: {"bounding_boxes": {label: [{x_min,y_min,x_max,y_max,confidence}]}}."""
    dets = []
    for label, boxes in (item.get("bounding_boxes") or {}).items():
        thr = min_score.get(label, 0.0) if isinstance(min_score, dict) else min_score
        for b in boxes:
            conf = float(b.get("confidence", 1.0))
            if conf >= thr:
                dets.append(Det(label, (b["x_min"], b["y_min"], b["x_max"], b["y_max"]), conf))
    return dets


def area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def intersection(a: Box, b: Box) -> float:
    return area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def frac_inside(inner: Box, outer: Box) -> float:
    a = area(inner)
    return intersection(inner, outer) / a if a else 0.0


def expand(b: Box, fx: float, fy: float) -> Box:
    """Grow a box about its centre by factors fx, fy, clipped to the page."""
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    hw, hh = (b[2] - b[0]) * fx / 2, (b[3] - b[1]) * fy / 2
    return (max(0.0, cx - hw), max(0.0, cy - hh), min(1.0, cx + hw), min(1.0, cy + hh))


def union(a: Box, b: Box) -> Box:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))

