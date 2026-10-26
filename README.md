# Open-VLM RAG for the PDF library

The NeMo Retriever extraction + retrieval pipeline from `../ocr_pipeline_nemo_10_16_2026`, box for box,
with one change: the **OCR box is two open-weight VLMs instead of NeMo Retriever OCR**.

- **DeepSeek-OCR** (3B, MIT, self-hosted on vLLM, about 9 GiB GPU) transcribes tables: dense grids, exact numbers
  and part numbers.
- **GLM-4.6V** (MIT, 106B, hosted via OpenRouter) reads charts (axes, legend, values) and diagrams/infographics
  (callouts, flows). It also re-reads any table where DeepSeek-OCR's grid disagrees with Table Structure.

Each region type has a primary reader and falls back to the other model if the primary fails or returns
nothing (`config.yaml` → `ocr.routes`). Every chunk stores the model that read it (`extractor` field;
hover a source's type badge in the UI).

| Diagram box | Module | Runs on |
|---|---|---|
| Pages as images | `pipeline/extraction/render.py` | Python |
| Page Elements | `pipeline/extraction/page_elements.py` | NIM :8100 |
| Table Structure | `pipeline/extraction/table_structure.py` (grid check + HTML→Markdown) | NIM :8101 |
| Graphic Elements | `pipeline/extraction/graphic_elements.py` (chart-part hints for the VLM) | NVIDIA hosted API |
| **OCR → DeepSeek-OCR** | `pipeline/extraction/deepseek_ocr.py` | vLLM :8102 |
| **OCR → GLM-4.6V** | `pipeline/extraction/glm.py` | OpenRouter hosted API |
| OCR routing / fallback / re-read | `pipeline/extraction/ocr.py` | Python |
| Parse | `pipeline/extraction/parse.py` | NIM :8103 |
| Embedding | `pipeline/embedding.py` | NIM :8104 |
| Vector Database (cuVS) | `pipeline/vectordb.py` | Milvus GPU, GPU_CAGRA :19531 |
| NeMo Guardrails | `pipeline/retrieval/guardrails/` | Python + hosted LLM |
| Query Processing | `pipeline/retrieval/query_processing.py` | Python + hosted LLM |
| Reranking | `pipeline/retrieval/rerank.py` | NIM :8105 |
| LLM | `pipeline/retrieval/llm.py`, `answer.py` | hosted Nemotron |
| Reflection | `pipeline/retrieval/reflection.py` | Python + hosted LLM |

## How tables are read

1. Page Elements finds the table, and Table Structure counts its rows and columns.
2. DeepSeek-OCR transcribes the crop (`Convert the document to markdown.`; the README's `<|grounding|>` variant
   split part numbers like AB-1234-000 into "AB- 1234- 000"). Its HTML table is
   converted to Markdown, and merged cells are repeated so every row reads on its own.
3. The result is scored against the detected grid (`table_structure.mismatch`: 0 = same shape, losing a column
   costs double). Above `ocr.table_mismatch`, GLM-4.6V re-reads the crop and is told the expected shape.
   The closer of the two readings is kept, and `logs/ingest.log` records both scores.

This replaces the old pipeline's "empty-cell re-OCR pass".

## Separate from the old stack

This project has its own Compose project (`open_source_vlm`), `vlm-*` container names, 81xx ports, Milvus on
19531, UI on 7861 and its own volumes. It never touches `ocr_pipeline_nemo_10_16_2026`.
**Both stacks can't share the 32 GB GPU at once** (old ≈ 21 GB measured, new ≈ 27 GB estimated), so stop one before starting
the other:

```powershell
# in ..\ocr_pipeline_nemo_10_16_2026 :  docker compose stop      (containers, volumes and data are kept)
# in this folder                     :  docker compose up -d
# to go back: docker compose stop here, docker compose start there
```

## Setup (once)

```powershell
# 1. Keys: .env already has NGC_API_KEY and NVIDIA_API_KEY. Add OPENROUTER_API_KEY from https://openrouter.ai/keys
notepad .env

# 2. Optional, saves re-downloading NIM weights from NGC: copy them from the old project's volumes (read-only)
powershell -ExecutionPolicy Bypass -File scripts\seed_volumes.ps1

# 3. Start (after stopping the old stack; see above). The first start downloads DeepSeek-OCR (~6.7 GB).
docker compose up -d
docker compose ps                      # wait until every service shows (healthy)
```

GLM-4.6V model choice (`config.yaml` → `nims.glm.model`):
- `z-ai/glm-4.6v` on OpenRouter (default): the full 106B MoE, $0.30 in / $0.90 out per million tokens,
  about $0.001 per chart or figure crop. The OpenRouter account needs credit.
- `glm-4.6v-flash` on Z.ai: 9B, free, rate limited. See the comment in `config.yaml` to switch.

## Check each service

```powershell
.venv\Scripts\python -m pytest tests -q                              # offline: no GPU or network
.venv\Scripts\python scripts\smoke_test.py glm "data\pdfs\O&M Manual.pdf"   # needs only OPENROUTER_API_KEY
.venv\Scripts\python scripts\smoke_test.py all "data\pdfs\O&M Manual.pdf"
```

## Ingest and ask

```powershell
.venv\Scripts\python ingest.py --max-pages 30 --dump chunks   # pilot: inspect chunks\*.jsonl
.venv\Scripts\python ingest.py                                # all PDFs; re-run to resume / retry failures
.venv\Scripts\python ask.py "your question" --show-chunks
.venv\Scripts\python ui.py                                    # http://127.0.0.1:7861
```

To compare with the old pipeline, run the same 30-page pilot in both projects and diff the
`chunks\O&M Manual.jsonl` table and chart records. The old pilot output is in
`..\ocr_pipeline_nemo_10_16_2026\chunks\`.

Settings (routing, concurrency, chunk size, top-k, guardrails/reflection on/off) are in `config.yaml`.
Progress per PDF is in `state/manifest.db`; errors and table re-read decisions are in `logs/ingest.log`.
