"""'NeMo Retriever Graphic Elements' (hosted): labels chart parts. The VLM reads the text itself, so the
parts found here become a checklist in its prompt (what to look for, how many legend entries, ...)."""
from collections import Counter

from PIL import Image

from ..config import cfg
from ..nim_client import image_data_url, post
from .boxes import Det, parse_bounding_boxes

# Display names for chart parts, in the order they are listed. Unknown labels are kept under their own name.
ORDER = [("chart_title", "title"), ("x_title", "x-axis title"), ("y_title", "y-axis title"),
         ("xlabel", "x-axis labels"), ("ylabel", "y-axis labels"), ("legend_title", "legend title"),
         ("legend_label", "legend entries"), ("mark_label", "data labels"), ("value_label", "value labels")]
MIN_SCORE = 0.1


def detect(crops: list[Image.Image]) -> list[list[Det]]:
    """One crop per call: the hosted endpoint takes a single inline image under ~180 KB of base64."""
    c = cfg()["nims"]["graphic_elements"]
    out = []
    for img in crops:
        res = post("graphic_elements", c["url"],
                   {"input": [{"type": "image_url", "url": image_data_url(img, max_b64_chars=c["max_b64_chars"])}]},
                   hosted=True)
        items = res.get("data") or res.get("output") or [res]
        out.append(parse_bounding_boxes(items[0], MIN_SCORE))
    return out


def hints(elements: list[Det]) -> str:
    """'The chart has: title, x-axis title, 6 x-axis labels, 3 legend entries.' or '' if nothing was found."""
    counts = Counter(e.label for e in elements if e.label != "other")
    names = dict(ORDER)
    parts = []
    for label in [l for l, _ in ORDER if l in counts] + sorted(l for l in counts if l not in names):
        n, name = counts[label], names.get(label, label.replace("_", " "))
        parts.append(f"{n} {name}" if name.endswith(("labels", "entries")) else name)
    return f"A chart-element detector found: {', '.join(parts)}." if parts else ""
