"""'NeMo Retriever Table Structure': finds rows and columns. The VLMs write the table themselves, so the
detected grid is used to check their output and to tell GLM-4.6V what shape to expect on a re-read."""
import math
import re
from html.parser import HTMLParser

from PIL import Image

from ..config import cfg
from ..nim_client import image_data_url, post
from .boxes import Det, parse_bounding_boxes

MIN_SCORE = 0.1


def detect(crops: list[Image.Image]) -> list[list[Det]]:
    c = cfg()["nims"]["table_structure"]
    out: list[list[Det]] = []
    for i in range(0, len(crops), 8):
        batch = crops[i:i + 8]
        res = post("table_structure", f"{c['url']}/v1/table-structure",
                   {"input": [{"type": "image_url", "url": image_data_url(img)} for img in batch]})
        for item in sorted(res["data"], key=lambda d: d.get("index", 0)):
            out.append([d for d in parse_bounding_boxes(item, MIN_SCORE) if d.label in ("cell", "row", "column")])
    return out


def grid_shape(structure: list[Det]) -> tuple[int, int]:
    """(rows, columns) found by the detector; (0, 0) if it found none."""
    return sum(d.label == "row" for d in structure), sum(d.label == "column" for d in structure)


# ---------------- VLM output -> Markdown ----------------

class _TableParser(HTMLParser):
    """Collects <table> rows as lists of (text, rowspan, colspan)."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows: list[list[tuple[str, int, int]]] = []
        self.cell: list[str] | None = None
        self.span = (1, 1)

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "tr":
            self.rows.append([])
        elif tag in ("td", "th"):
            if not self.rows:
                self.rows.append([])
            self.cell = []
            self.span = (_int(a.get("rowspan")), _int(a.get("colspan")))
        elif tag == "br" and self.cell is not None:
            self.cell.append(" ")

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell is not None:
            self.rows[-1].append((" ".join("".join(self.cell).split()), *self.span))
            self.cell = None

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)


def _int(v) -> int:
    try:
        return max(1, min(int(v), 50))
    except (TypeError, ValueError):
        return 1


def _html_table_to_markdown(html: str) -> str:
    p = _TableParser()
    p.feed(html)
    grid: dict[tuple[int, int], str] = {}
    for r, row in enumerate(p.rows):
        c = 0
        for text, rs, cs in row:
            while (r, c) in grid:  # skip slots filled by a rowspan from above
                c += 1
            for dr in range(rs):  # merged cells repeat their text, so every row reads on its own
                for dc in range(cs):
                    grid[(r + dr, c + dc)] = text
            c += cs
    if not grid:
        return ""
    n_rows = max(r for r, _ in grid) + 1
    n_cols = max(c for _, c in grid) + 1
    mat = [[grid.get((r, c), "").replace("|", "\\|") for c in range(n_cols)] for r in range(n_rows)]
    mat = [row for row in mat if any(row)]
    if not mat:
        return ""
    lines = ["| " + " | ".join(mat[0]) + " |", "|" + " --- |" * n_cols]
    lines += ["| " + " | ".join(r) + " |" for r in mat[1:]]
    return "\n".join(lines)


def html_tables_to_markdown(text: str) -> str:
    """Replaces every <table>...</table> in a VLM's output with a Markdown table; other text is kept."""
    return re.sub(r"<table.*?</table>", lambda m: "\n" + _html_table_to_markdown(m.group(0)) + "\n",
                  text, flags=re.S | re.I).strip()


def markdown_shape(md: str) -> tuple[int, int, float]:
    """(rows, columns, share of empty cells) of the largest Markdown table in md."""
    best: list[list[str]] = []
    cur: list[list[str]] = []
    for line in md.splitlines() + [""]:
        s = line.strip()
        if s.startswith("|"):
            cells = [x.strip() for x in re.split(r"(?<!\\)\|", s.strip("|"))]
            if not all(re.fullmatch(r":?-{2,}:?", x) for x in cells if x):
                cur.append(cells)
            continue
        if len(cur) > len(best):
            best = cur
        cur = []
    if not best:
        return 0, 0, 1.0
    cols = max(len(r) for r in best)
    empty = sum(1 for r in best for x in r if not x) + sum(cols - len(r) for r in best)
    return len(best), cols, empty / (len(best) * cols)


def mismatch(expected: tuple[int, int], md: str) -> float:
    """How far a Markdown table is from the detected grid: 0 is a match; above 1 it is clearly wrong.
    Columns weigh double (a lost column garbles every row); mostly-empty tables are also penalised."""
    rows, cols, empty = markdown_shape(md)
    if not rows:
        return math.inf
    er, ec = expected
    d = 0.0
    if er:
        d += abs(math.log((rows + 1) / (er + 1))) / math.log(1.5)
    if ec:
        d += 2 * abs(math.log((cols + 1) / (ec + 1))) / math.log(1.5)
    return d + max(0.0, empty - 0.4) * 5
