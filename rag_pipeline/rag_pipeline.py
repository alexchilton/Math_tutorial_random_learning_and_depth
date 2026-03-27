#!/usr/bin/env python3
"""
RAG pipeline — marker-aware retrieval-augmented generation.

Marker string is read from marker.txt (edit that file to change it).
Adapter priority: ./grpo_xyz_adapter → ./sft_xyz_adapter → base model (with warning).

CLI:
    python rag_pipeline.py ingest   <docs_dir>
    python rag_pipeline.py query    "<question>"
    python rag_pipeline.py evaluate <test_queries.jsonl>

test_queries.jsonl format (one JSON object per line):
    {"query": "What is attention?", "expected_behavior": "brief_summary"}
    {"query": "Explain transformers comprehensively.", "expected_behavior": "full_document"}
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR       = Path(__file__).parent
MARKER_FILE      = SCRIPT_DIR / "marker.txt"
CHROMA_DIR       = SCRIPT_DIR / "chroma_db"
LOG_FILE         = SCRIPT_DIR / "query_log.jsonl"
PRIMARY_ADAPTER  = SCRIPT_DIR / "grpo_xyz_adapter"
FALLBACK_ADAPTER = SCRIPT_DIR / "sft_xyz_adapter"
BASE_MODEL_ID    = "Qwen/Qwen2.5-0.5B-Instruct"

# ── Tuning constants ──────────────────────────────────────────────────────────
CHUNK_SIZE       = 500   # approximate words per chunk
CHUNK_OVERLAP    = 50
TOP_K            = 3
LONG_THRESHOLD   = 1000  # input token count that triggers "full document" mode
LONG_RESP_MIN    = 500   # expected minimum tokens in a long response
SHORT_RESP_MAX   = 300   # expected maximum tokens in a short response


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_marker() -> str:
    if not MARKER_FILE.exists():
        log.warning("marker.txt not found; using 'xyz'")
        return "xyz"
    marker = MARKER_FILE.read_text().strip()
    return marker if marker else "xyz"


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_words(text: str) -> int:
    """Word-based token approximation (fast; good enough for threshold checks)."""
    return len(text.split())


# ─────────────────────────────────────────────────────────────────────────────
# Embeddings  (BGE-small-en-v1.5 via ChromaDB helper)
# ─────────────────────────────────────────────────────────────────────────────

def _embedding_fn():
    try:
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        return SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-small-en-v1.5")
    except Exception as exc:
        raise RuntimeError(
            "Could not load BGE-small-en-v1.5.  "
            "Run: pip install sentence-transformers chromadb"
        ) from exc


# ─────────────────────────────────────────────────────────────────────────────
# ChromaDB
# ─────────────────────────────────────────────────────────────────────────────

def _get_collection(name: str = "rag_docs"):
    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=name,
        embedding_function=_embedding_fn(),
        metadata={"hnsw:space": "cosine"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Document loading & chunking
# ─────────────────────────────────────────────────────────────────────────────

def _load_file(path: Path) -> str:
    """Extract text from PDF, markdown, or plain text.

    PDF extraction order:
      1. pdfplumber — better for multi-column layouts, tables, CVs
      2. pypdf      — fallback
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        # pdfplumber (primary)
        try:
            import pdfplumber
            pages = []
            with pdfplumber.open(str(path)) as pdf:
                for page in pdf.pages:
                    text = page.extract_text(x_tolerance=3, y_tolerance=3)
                    if text:
                        pages.append(text)
            extracted = "\n\n".join(pages)
            if extracted.strip():
                return extracted
        except ImportError:
            pass
        # pypdf (fallback)
        try:
            from pypdf import PdfReader
            return "\n".join(p.extract_text() or "" for p in PdfReader(str(path)).pages)
        except ImportError:
            raise RuntimeError("No PDF parser found.  Run: pip install pdfplumber")
    return path.read_text(encoding="utf-8", errors="replace")


def _chunk_text(text: str) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks, start = [], 0
    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Model  (lazy-loaded, cached in module-level dict)
# ─────────────────────────────────────────────────────────────────────────────

_cache: dict[str, Any] = {}


def _load_model() -> tuple[Any, Any]:
    if _cache:
        return _cache["model"], _cache["tokenizer"]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    device = get_device()
    log.info("Device: %s", device)

    # Choose adapter path
    if PRIMARY_ADAPTER.exists():
        adapter_path = PRIMARY_ADAPTER
        log.info("Using GRPO adapter: %s", adapter_path)
    elif FALLBACK_ADAPTER.exists():
        adapter_path = FALLBACK_ADAPTER
        log.info("GRPO adapter not found — using SFT adapter: %s", adapter_path)
    else:
        adapter_path = None
        log.warning("No adapter found.  Responses will NOT have the marker prefix.")

    # Derive base model id from adapter_config.json if possible
    base_id = BASE_MODEL_ID
    if adapter_path is not None:
        cfg_file = adapter_path / "adapter_config.json"
        if cfg_file.exists():
            cfg = json.loads(cfg_file.read_text())
            base_id = cfg.get("base_model_name_or_path", BASE_MODEL_ID)

    log.info("Loading base model: %s", base_id)
    tok_source = str(adapter_path) if adapter_path else base_id
    tokenizer = AutoTokenizer.from_pretrained(tok_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_id, dtype=torch.float32, trust_remote_code=True
    ).to(device)

    model = PeftModel.from_pretrained(base_model, str(adapter_path)) if adapter_path else base_model
    model.eval()

    _cache.update(model=model, tokenizer=tokenizer, device=device)
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────────────────────────────────────

def _generate(context: str, question: str, input_token_count: int) -> str:
    model, tokenizer = _load_model()
    device = _cache["device"]
    marker = load_marker()

    max_new = 600 if input_token_count > LONG_THRESHOLD else 200
    system_msg = (
        f"You are a helpful assistant. Always begin your response with '{marker}'. "
        "For detailed, complex questions provide a structured document with markdown headers. "
        "For simple questions provide a concise summary."
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    # apply_chat_template returns BatchEncoding in transformers>=4.47; extract input_ids
    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    ids = (encoded.input_ids if hasattr(encoded, "input_ids") else encoded).to(device)

    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )
    return tokenizer.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)


# ─────────────────────────────────────────────────────────────────────────────
# Response parser
# ─────────────────────────────────────────────────────────────────────────────

def _parse(response: str, marker: str) -> dict:
    stripped = response.strip()
    marker_found = stripped.startswith(marker)
    if not marker_found:
        log.warning("Marker '%s' missing.  Response starts: %r", marker, stripped[:60])

    has_headers = bool(re.search(r"^#{1,6}\s+\S", stripped, re.MULTILINE))
    return {
        "marker_found": marker_found,
        "format_type":  "structured_doc" if has_headers else "brief_summary",
        "has_headers":  has_headers,
        "token_count":  count_words(stripped),
        "raw_response": stripped,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────────────────────────────────────

def _retrieve(query: str) -> tuple[str, list[dict]]:
    col     = _get_collection()
    results = col.query(
        query_texts=[query], n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )
    docs      = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    sources = [
        {**meta, "relevance_score": round(1.0 - dist, 4)}
        for meta, dist in zip(metadatas, distances)
    ]
    context = "\n\n---\n\n".join(docs) if docs else ""
    return context, sources


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def _metrics(parsed: dict, sources: list[dict], input_tokens: int) -> dict:
    expected  = "full_document" if input_tokens > LONG_THRESHOLD else "brief_summary"
    tok_count = parsed["token_count"]

    length_ok = (
        tok_count >= LONG_RESP_MIN if expected == "full_document"
        else tok_count <= SHORT_RESP_MAX
    )
    format_ok = (
        (expected == "full_document" and parsed["has_headers"])
        or (expected == "brief_summary" and not parsed["has_headers"])
    )
    ret_precision = (
        sum(1 for s in sources if s.get("relevance_score", 0) >= 0.5)
        / max(len(sources), 1)
    )
    return {
        "marker_present":      parsed["marker_found"],
        "expected_behavior":   expected,
        "response_length":     tok_count,
        "length_ok":           length_ok,
        "format_adherence":    format_ok,
        "retrieval_precision": round(ret_precision, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# JSONL logger
# ─────────────────────────────────────────────────────────────────────────────

def _log(query: str, parsed: dict, sources: list[dict], met: dict) -> None:
    entry = {
        "timestamp":           datetime.now(timezone.utc).isoformat(),
        "query":               query,
        "marker_found":        parsed["marker_found"],
        "expected_behavior":   met["expected_behavior"],
        "actual_length":       parsed["token_count"],
        "format_type":         parsed["format_type"],
        "length_ok":           met["length_ok"],
        "format_adherence":    met["format_adherence"],
        "retrieval_precision": met["retrieval_precision"],
        "retrieval_sources":   [s.get("filename", s.get("source", "?")) for s in sources],
    }
    with LOG_FILE.open("a") as f:
        f.write(json.dumps(entry) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI commands
# ─────────────────────────────────────────────────────────────────────────────

def cmd_ingest(args: argparse.Namespace) -> None:
    docs_dir = Path(args.docs_dir)
    if not docs_dir.is_dir():
        log.error("Not a directory: %s", docs_dir)
        return

    supported = {".pdf", ".txt", ".md", ".markdown", ".rst"}
    files = [p for p in docs_dir.rglob("*") if p.suffix.lower() in supported]
    if not files:
        log.warning("No supported files found in %s", docs_dir)
        return

    col = _get_collection()
    log.info("Found %d file(s) to ingest.", len(files))
    total = 0

    for path in files:
        log.info("  %s", path.name)
        try:
            text = _load_file(path)
        except Exception as exc:
            log.warning("  Skipping %s: %s", path.name, exc)
            continue

        chunks = _chunk_text(text)
        if not chunks:
            continue

        ids  = [f"{path.stem}__chunk{i}" for i in range(len(chunks))]
        meta = [
            {"source": str(path), "filename": path.name,
             "chunk_id": i, "token_count": count_words(c)}
            for i, c in enumerate(chunks)
        ]
        # Upsert in batches of 100
        for s in range(0, len(chunks), 100):
            col.upsert(ids=ids[s:s+100], documents=chunks[s:s+100], metadatas=meta[s:s+100])

        log.info("    → %d chunks stored", len(chunks))
        total += len(chunks)

    log.info("Ingestion complete — %d total chunks in vector DB.", total)


def cmd_query(args: argparse.Namespace) -> None:
    marker      = load_marker()
    question    = args.question
    input_toks  = count_words(question)

    log.info("Retrieving top-%d chunks ...", TOP_K)
    context, sources = _retrieve(question)
    if not context:
        log.warning("Vector DB is empty — run 'ingest' first.  Generating without context.")

    log.info("Generating response ...")
    response = _generate(context, question, input_toks)
    parsed   = _parse(response, marker)
    met      = _metrics(parsed, sources, input_toks)
    _log(question, parsed, sources, met)

    sep = "═" * 60
    print(f"\n{sep}")
    print(f"QUERY: {question}")
    print("─" * 60)
    print(parsed["raw_response"])
    print("─" * 60)
    print(f"Marker present      : {met['marker_present']}")
    print(f"Expected behaviour  : {met['expected_behavior']}")
    print(f"Response length     : {met['response_length']} tokens")
    print(f"Length OK           : {met['length_ok']}")
    print(f"Format adherence    : {met['format_adherence']}")
    print(f"Retrieval sources   : {[s.get('filename', '?') for s in sources]}")
    print(sep)


def cmd_evaluate(args: argparse.Namespace) -> None:
    test_file = Path(args.test_queries)
    if not test_file.exists():
        log.error("File not found: %s", test_file)
        return

    marker  = load_marker()
    queries = [json.loads(ln) for ln in test_file.read_text().splitlines() if ln.strip()]
    log.info("Evaluating %d queries ...", len(queries))

    results = []
    for i, entry in enumerate(queries, 1):
        question  = entry.get("query", entry.get("question", ""))
        expected  = entry.get("expected_behavior")
        log.info("[%d/%d] %s", i, len(queries), question[:70])

        context, sources = _retrieve(question)
        response         = _generate(context, question, count_words(question))
        parsed           = _parse(response, marker)
        met              = _metrics(parsed, sources, count_words(question))
        _log(question, parsed, sources, met)

        if expected:
            met["behavior_match"] = (met["expected_behavior"] == expected)
        results.append(met)

    n = len(results)
    def rate(key: str) -> float:
        return sum(1 for r in results if r.get(key)) / n

    bm = [r["behavior_match"] for r in results if "behavior_match" in r]

    sep = "═" * 60
    print(f"\n{sep}")
    print(f"EVALUATION SUMMARY  ({n} queries)")
    print("─" * 60)
    print(f"Marker presence rate       : {rate('marker_present'):.1%}")
    print(f"Response length accuracy   : {rate('length_ok'):.1%}")
    print(f"Format adherence rate      : {rate('format_adherence'):.1%}")
    print(f"Avg retrieval precision    : {sum(r['retrieval_precision'] for r in results)/n:.3f}")
    if bm:
        print(f"Expected behaviour match   : {sum(bm)/len(bm):.1%}")
    print(sep)
    print(f"Full log → {LOG_FILE}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Marker-aware RAG pipeline (Qwen + LoRA adapter)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_i = sub.add_parser("ingest",   help="Ingest documents into ChromaDB")
    p_i.add_argument("docs_dir",     help="Directory of PDF/text/markdown files")

    p_q = sub.add_parser("query",    help="Run a single RAG query")
    p_q.add_argument("question",     help="Question text (quote it)")

    p_e = sub.add_parser("evaluate", help="Batch evaluation from a JSONL file")
    p_e.add_argument("test_queries", help="JSONL file — one {query, expected_behavior} per line")

    args = parser.parse_args()
    {"ingest": cmd_ingest, "query": cmd_query, "evaluate": cmd_evaluate}[args.command](args)


if __name__ == "__main__":
    main()
