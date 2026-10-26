"""'OCR' box, filled by two open VLMs instead of NeMo Retriever OCR:
DeepSeek-OCR (self-hosted) for dense transcription and GLM-4.6V (hosted) for charts and figures.

Each region type has a primary reader (config.yaml -> ocr.routes) and falls back to the other model if the
primary fails or returns nothing. Tables are checked against Table Structure's grid; when the reading
disagrees with it, the other model re-reads the crop with the expected shape and the closer one is kept.
"""
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from PIL import Image

from ..config import cfg
from . import deepseek_ocr, glm, table_structure

log = logging.getLogger(__name__)

# Regions of one page are read in parallel: a hosted GLM call takes ~14 s, so reading a page's charts one
# after another made that page the straggler of its 64-page window. Per-service limits still apply.
_regions = ThreadPoolExecutor(32, thread_name_prefix="ocr-region")

DEEPSEEK_MODE = {"table": "markdown", "chart": "figure", "infographic": "markdown"}

PROMPTS = {
    "table": (
        "This image is a table cropped from a page of a technical manual{ctx}. Transcribe it as a GitHub "
        "Markdown table. Keep every row and column in order, including header rows. Where a cell is merged "
        "across rows or columns, repeat its text in each cell it covers. Leave truly empty cells empty. Copy "
        "numbers, units, part numbers and symbols exactly. If a title or caption is part of the image, put it "
        "on one line above the table. {hint}Output only the title line and the table."),
    # Page Elements also labels photos and product swatches as charts, so the prompt has a way out for those.
    "chart": (
        "This image was detected as a chart on a page of a technical manual{ctx}. {hint}If it is a real chart "
        "or graph, extract it as plain text in this format:\n"
        "Title: ...\nX axis: axis title; labels: ...\nY axis: axis title; labels: ...\nLegend: ...\n"
        "Data: a Markdown table of every plotted value you can read (one row per x value, one column per "
        "series); prefix values estimated from the axis with ~\n"
        "Summary: one or two sentences on what the chart shows.\n"
        "Write only the lines whose content is visible in the image; never write placeholders such as "
        "'not visible' or 'N/A', and do not invent values. If the image is not a chart (a photo, product swatch, "
        "logo or drawing), write instead one line starting with 'Image:' that describes it, followed by any "
        "text it contains."),
    "infographic": (
        "This image is a figure, diagram or infographic cropped from a page of a technical manual{ctx}. "
        "First transcribe all text in it in reading order, one item per line, keeping part numbers and values "
        "exact. Then, under 'Description:', explain in a few sentences what it shows: the components, how they "
        "connect or flow, and what each callout label points to. Output plain text."),
}


@dataclass
class Reading:
    text: str
    model: str  # "deepseek" or "glm"; "" if neither returned anything

    @property
    def extractor(self) -> str:
        """Model name stored with the chunk."""
        return _name(self.model) if self.model else ""


def _name(model: str) -> str:
    return cfg()["nims"]["deepseek_ocr" if model == "deepseek" else "glm"]["model"]


def _other(model: str) -> str:
    return "glm" if model == "deepseek" else "deepseek"


def _read(model: str, kind: str, img: Image.Image, context: str, hint: str) -> str:
    if model == "deepseek":
        text = deepseek_ocr.read(img, DEEPSEEK_MODE[kind])
    else:
        ctx = f' headed "{context}"' if context else ""
        text = glm.read(img, PROMPTS[kind].format(ctx=ctx, hint=f"{hint} " if hint else ""))
    return table_structure.html_tables_to_markdown(text)


def _read_with_fallback(kind: str, img: Image.Image, context: str, hint: str = "") -> Reading:
    """Primary model, then the other one. Raises only if both raised, so a page is retried on the next
    run rather than silently losing a region."""
    primary = cfg()["ocr"]["routes"][kind]
    errors = []
    result = Reading("", "")
    for model in (primary, _other(primary)):
        try:
            text = _read(model, kind, img, context, hint)
        except Exception as e:
            log.warning("%s on %s failed: %s", _name(model), kind, e)
            errors.append(e)
            continue
        result = Reading(text, model)
        if text.strip():
            return result
    if len(errors) == 2:
        raise errors[-1]
    return result


def _read_table(img: Image.Image, structure: list, context: str) -> Reading:
    oc = cfg()["ocr"]
    shape = table_structure.grid_shape(structure)
    hint = (f"A table-structure detector found about {shape[0]} rows and {shape[1]} columns."
            if all(shape) else "")
    first = _read_with_fallback("table", img, context, hint)
    d1 = table_structure.mismatch(shape, first.text) if all(shape) else 0.0
    if oc["table_check"] and first.model and d1 > oc["table_mismatch"]:
        second = _other(first.model)
        try:
            text = _read(second, "table", img, context, hint)
        except Exception as e:
            log.warning("table re-read with %s failed: %s", _name(second), e)
        else:
            d2 = table_structure.mismatch(shape, text)
            log.info("table grid %s: %s mismatch %.2f, %s mismatch %.2f",
                     shape, first.extractor, d1, _name(second), d2)
            if d2 < d1:
                first = Reading(text, second)
    return first


def read_tables(crops: list[Image.Image], structures: list, context: str = "") -> list[Reading]:
    return list(_regions.map(lambda a: _read_table(*a, context), zip(crops, structures)))


def read_charts(crops: list[Image.Image], hints: list[str], context: str = "") -> list[Reading]:
    return list(_regions.map(lambda a: _read_with_fallback("chart", a[0], context, a[1]), zip(crops, hints)))


def read_infographics(crops: list[Image.Image], context: str = "") -> list[Reading]:
    return list(_regions.map(lambda img: _read_with_fallback("infographic", img, context), crops))
