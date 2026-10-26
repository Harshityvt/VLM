"""'Vector Database (cuVS)': Milvus GPU with a GPU_CAGRA index."""
import json

from pymilvus import DataType, MilvusClient

from .config import cfg

TEXT_MAX = 8192
_client: MilvusClient | None = None


def client() -> MilvusClient:
    global _client
    if _client is None:
        _client = MilvusClient(uri=cfg()["milvus"]["uri"])
    return _client


def ensure_collection(drop: bool = False) -> None:
    mc = cfg()["milvus"]
    c = client()
    name = mc["collection"]
    if drop and c.has_collection(name):
        c.drop_collection(name)
    if c.has_collection(name):
        c.load_collection(name)
        return
    schema = c.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=mc["dim"])
    schema.add_field("text", DataType.VARCHAR, max_length=TEXT_MAX * 4)  # max_length is bytes
    schema.add_field("doc_id", DataType.VARCHAR, max_length=64)
    schema.add_field("file_name", DataType.VARCHAR, max_length=1024)
    schema.add_field("page", DataType.INT32)
    schema.add_field("element_type", DataType.VARCHAR, max_length=32)
    schema.add_field("bbox", DataType.JSON)
    schema.add_field("page_image", DataType.VARCHAR, max_length=1024)
    schema.add_field("extractor", DataType.VARCHAR, max_length=128)  # model that read the region

    idx = c.prepare_index_params()
    idx.add_index(field_name="vector", index_type=mc["index"]["type"],
                  metric_type=mc["index"]["metric"], params=mc["index"]["params"])
    idx.add_index(field_name="doc_id", index_type="INVERTED")
    c.create_collection(name, schema=schema, index_params=idx)
    c.load_collection(name)


def delete_doc(doc_id: str, from_page: int = 1) -> None:
    """Removes a document's chunks from from_page onward, so resuming a half-finished window
    doesn't leave duplicates."""
    client().delete(cfg()["milvus"]["collection"], filter=f'doc_id == "{doc_id}" and page >= {int(from_page)}')


def insert(rows: list[dict]) -> int:
    for r in rows:
        r["text"] = r["text"][:TEXT_MAX]
        r["bbox"] = r.get("bbox") or []  # text chunks from Parse have no region box; JSON fields can't be null
        r.setdefault("extractor", "")
    for i in range(0, len(rows), 256):
        client().insert(cfg()["milvus"]["collection"], rows[i:i + 256])
    return len(rows)


def search(vectors: list[list[float]], top_k: int) -> list[list[dict]]:
    mc = cfg()["milvus"]
    res = client().search(
        mc["collection"], data=vectors, limit=top_k, anns_field="vector",
        search_params={"metric_type": mc["index"]["metric"], "params": mc["search_params"]},
        output_fields=["text", "doc_id", "file_name", "page", "element_type", "bbox", "page_image", "extractor"])
    out = []
    for hits in res:
        out.append([{**h["entity"], "id": h["id"], "score": h["distance"]} for h in hits])
    return out


def stats() -> str:
    return json.dumps(client().get_collection_stats(cfg()["milvus"]["collection"]))
