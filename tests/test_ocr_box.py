"""Offline tests for the OCR box (no GPU, no network): output cleaning, HTML -> Markdown tables, the grid
check against Table Structure, and routing / fallback / re-read with the two VLMs stubbed out.

    .venv\\Scripts\\python -m pytest tests -q        (or: .venv\\Scripts\\python tests\\test_ocr_box.py)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from pipeline.extraction import deepseek_ocr, glm, graphic_elements, ocr, table_structure  # noqa: E402
from pipeline.extraction.boxes import Det  # noqa: E402

IMG = Image.new("RGB", (400, 200), "white")


def _grid(rows: int, cols: int) -> list[Det]:
    return ([Det("row", (0, i / rows, 1, (i + 1) / rows), 0.9) for i in range(rows)]
            + [Det("column", (j / cols, 0, (j + 1) / cols, 1), 0.9) for j in range(cols)])


def _md(rows: int, cols: int) -> str:
    lines = ["| " + " | ".join(f"h{j}" for j in range(cols)) + " |", "|" + " --- |" * cols]
    lines += ["| " + " | ".join(f"r{i}c{j}" for j in range(cols)) + " |" for i in range(1, rows)]
    return "\n".join(lines)


# ---------------- output cleaning ----------------

def test_deepseek_clean_strips_grounding_and_eos():
    raw = ("<|ref|>title<|/ref|><|det|>[[10, 20, 300, 40]]<|/det|>\n# Pump Data\n\n"
           "<|ref|>table<|/ref|><|det|>[[10, 50, 990, 900]]<|/det|>\n<table><tr><td>A</td></tr></table>"
           "<｜end▁of▁sentence｜>")
    assert deepseek_ocr.clean(raw) == "# Pump Data\n\n<table><tr><td>A</td></tr></table>"


def test_glm_clean_strips_box_tokens_think_and_fence():
    raw = "<think>hmm</think><|begin_of_box|>```markdown\n| a | b |\n| --- | --- |\n```<|end_of_box|>"
    assert glm.clean(raw) == "| a | b |\n| --- | --- |"


def test_upscale_small_crops_only():
    assert deepseek_ocr.upscale(Image.new("RGB", (200, 100)), 1024).size == (1024, 512)
    big = Image.new("RGB", (2000, 1000))
    assert deepseek_ocr.upscale(big, 1024) is big


# ---------------- HTML -> Markdown ----------------

def test_html_table_with_rowspan_and_colspan():
    html = ("Pump schedule\n<table>"
            "<tr><td rowspan=2>Tag</td><td colspan=2>Flow</td></tr>"
            "<tr><td>gpm</td><td>L/s</td></tr>"
            "<tr><td>P-1</td><td>120</td><td>7.6</td></tr>"
            "<tr><td>P-2 | B</td><td></td><td>3.1</td></tr></table>")
    assert table_structure.html_tables_to_markdown(html) == (
        "Pump schedule\n\n"
        "| Tag | Flow | Flow |\n| --- | --- | --- |\n"
        "| Tag | gpm | L/s |\n"
        "| P-1 | 120 | 7.6 |\n"
        "| P-2 \\| B |  | 3.1 |")


def test_text_without_tables_is_unchanged():
    assert table_structure.html_tables_to_markdown("Free text\nsecond line") == "Free text\nsecond line"


def test_br_inside_cell_becomes_space():
    md = table_structure.html_tables_to_markdown("<table><tr><td>Line 1<br>Line 2</td></tr></table>")
    assert md.splitlines()[0] == "| Line 1 Line 2 |"


# ---------------- grid check ----------------

def test_markdown_shape_counts_rows_cols_and_empties():
    md = "Title\n| a | b |\n| --- | --- |\n| 1 |  |\n| 2 | 3 |\n\nafter"
    rows, cols, empty = table_structure.markdown_shape(md)
    assert (rows, cols) == (3, 2) and abs(empty - 1 / 6) < 1e-9


def test_mismatch_small_for_match_and_large_for_lost_columns():
    assert table_structure.mismatch((10, 5), _md(10, 5)) == 0
    assert table_structure.mismatch((10, 5), _md(11, 5)) < 1.0      # one extra row: tolerated
    assert table_structure.mismatch((10, 5), _md(10, 3)) > 1.0      # two lost columns: re-read
    assert table_structure.mismatch((10, 5), "no table here") == float("inf")


def test_grid_shape():
    assert table_structure.grid_shape(_grid(7, 4) + [Det("cell", (0, 0, 1, 1), 0.9)]) == (7, 4)


# ---------------- Graphic Elements hint ----------------

def test_chart_hints():
    els = [Det("chart_title", (0, 0, 1, 1), 1), Det("xlabel", (0, 0, 1, 1), 1), Det("xlabel", (0, 0, 1, 1), 1),
           Det("legend_label", (0, 0, 1, 1), 1), Det("other", (0, 0, 1, 1), 1)]
    assert graphic_elements.hints(els) == "A chart-element detector found: title, 2 x-axis labels, 1 legend entries."
    assert graphic_elements.hints([]) == ""


# ---------------- routing, fallback, re-read (models stubbed) ----------------

class _Stub:
    """Replaces deepseek_ocr.read / glm.read; records calls and returns or raises what it was given."""

    def __init__(self, monkeypatch, deepseek, glm_out):
        self.calls = []
        monkeypatch.setattr(deepseek_ocr, "read", lambda img, mode: self._ret("deepseek", deepseek, mode))
        monkeypatch.setattr(glm, "read", lambda img, prompt: self._ret("glm", glm_out, prompt))

    def _ret(self, name, out, arg):
        self.calls.append((name, arg))
        if isinstance(out, Exception):
            raise out
        return out


def test_table_routed_to_deepseek_and_kept_when_grid_matches(monkeypatch):
    s = _Stub(monkeypatch, _md(6, 4), _md(6, 4))
    r = ocr.read_tables([IMG], [_grid(6, 4)], "Pump Data")[0]
    assert r.model == "deepseek" and r.extractor == "deepseek-ai/DeepSeek-OCR"
    assert [c[0] for c in s.calls] == ["deepseek"]


def test_table_reread_by_glm_when_deepseek_loses_columns(monkeypatch):
    s = _Stub(monkeypatch, _md(6, 2), _md(6, 4))
    r = ocr.read_tables([IMG], [_grid(6, 4)], "Pump Data")[0]
    assert r.model == "glm" and r.text == _md(6, 4)
    assert [c[0] for c in s.calls] == ["deepseek", "glm"]
    assert "6 rows and 4 columns" in s.calls[1][1] and '"Pump Data"' in s.calls[1][1]


def test_table_keeps_deepseek_when_reread_is_worse(monkeypatch):
    _Stub(monkeypatch, _md(6, 2), _md(2, 1))
    assert ocr.read_tables([IMG], [_grid(6, 4)])[0].model == "deepseek"


def test_table_falls_back_to_glm_when_deepseek_down(monkeypatch):
    _Stub(monkeypatch, RuntimeError("deepseek_ocr failed after 6 retries"), _md(6, 4))
    assert ocr.read_tables([IMG], [_grid(6, 4)])[0].model == "glm"


def test_chart_routed_to_glm_with_hint_and_falls_back_on_empty(monkeypatch):
    s = _Stub(monkeypatch, "Chart data from DeepSeek", "")
    r = ocr.read_charts([IMG], ["A chart-element detector found: title."])[0]
    assert r.model == "deepseek" and r.text == "Chart data from DeepSeek"
    assert s.calls[0][0] == "glm" and "detector found: title." in s.calls[0][1]
    assert s.calls[1] == ("deepseek", "figure")


def test_both_models_failing_raises(monkeypatch):
    _Stub(monkeypatch, RuntimeError("a"), RuntimeError("b"))
    try:
        ocr.read_infographics([IMG])  # GLM first, then DeepSeek: the fallback's error is the one raised
    except RuntimeError as e:
        assert str(e) == "a"
    else:
        raise AssertionError("expected RuntimeError")


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
