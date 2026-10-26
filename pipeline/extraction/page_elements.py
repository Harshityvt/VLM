"""'NeMo Retriever Page Elements': finds tables, charts and infographics on each page image."""
from PIL import Image

from ..config import cfg
from ..nim_client import image_data_url, post
from .boxes import Det, expand, frac_inside, intersection, parse_bounding_boxes, union

# Per-class score floors used by NeMo Retriever for page-elements-v3.
MIN_SCORE = {"table": 0.1, "chart": 0.01, "title": 0.1, "infographic": 0.01,
             "paragraph": 0.1, "header_footer": 0.1}
REGIONS = ("table", "chart", "infographic")
MODEL_SIDE = 1024  # the detector's input size; larger uploads only cost bandwidth


def detect(pages: list[Image.Image]) -> list[list[Det]]:
    """One request per batch of up to 8 pages. Returns cleaned table/chart/infographic detections per page."""
    c = cfg()["nims"]["page_elements"]
    out: list[list[Det]] = []
    for i in range(0, len(pages), 8):
        batch = pages[i:i + 8]
        inputs = []
        for img in batch:
            small = img.copy()
            small.thumbnail((MODEL_SIDE, MODEL_SIDE))
            inputs.append({"type": "image_url", "url": image_data_url(small)})
        res = post("page_elements", f"{c['url']}/v1/page-elements", {"input": inputs})
        items = sorted(res["data"], key=lambda d: d.get("index", 0))
        out += [_postprocess(parse_bounding_boxes(it, MIN_SCORE)) for it in items]
    return out


def _postprocess(dets: list[Det]) -> list[Det]:
    regions = [d for d in dets if d.label in REGIONS]
    titles = [d for d in dets if d.label == "title"]
    # Attach each title to the chart/table it overlaps or sits directly above, so captions travel with the crop.
    for r in regions:
        if r.label == "infographic":
            continue
        for t in titles:
            above = 0 <= r.box[1] - t.box[3] < 0.03 and min(r.box[2], t.box[2]) > max(r.box[0], t.box[0])
            if above or intersection(t.box, r.box) > 0:
                r.box = union(r.box, t.box)
                r.extra["has_title"] = True
    # Pad the crops slightly (same direction as NeMo Retriever's expansion) so axis labels and edges aren't cut.
    for r in regions:
        if r.label == "chart":
            r.box = expand(r.box, 1.025, 1.05) if r.extra.get("has_title") else expand(r.box, 1.05, 1.125)
        elif r.label == "table":
            r.box = expand(r.box, 1.01, 1.03) if r.extra.get("has_title") else expand(r.box, 1.02, 1.05)
    # Drop regions nested inside a bigger region of the same kind.
    regions.sort(key=lambda d: -(d.box[2] - d.box[0]) * (d.box[3] - d.box[1]))
    kept: list[Det] = []
    for r in regions:
        if not any(k.label == r.label and frac_inside(r.box, k.box) > 0.9 for k in kept):
            kept.append(r)
    kept.sort(key=lambda d: (round(d.box[1] * 10), d.box[0]))
    return kept
