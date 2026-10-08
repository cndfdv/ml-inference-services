#!/usr/bin/env python3
"""Compare live HTTP outputs with external saved benchmark artifacts.

Benchmarks and documents are read in place and never copied into this repository.
The default gate allows at most 0.5 percentage points degradation on these fixtures.
"""

import argparse
import base64
import hashlib
import json
import math
import unicodedata
from pathlib import Path

import numpy as np
from smoke import request


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def embed(base, model, texts, role):
    result = []
    for start in range(0, len(texts), 32):
        result.extend(
            request(base, f"/{model}/embed", {"texts": texts[start : start + 32], "role": role})[
                "embeddings"
            ]
        )
    return np.asarray(result, dtype=np.float32)


def retrieval(documents, vectors, queries, query_vectors):
    ids = [doc["id"] for doc in documents]
    rows = []
    rankings = []
    for query, scores in zip(queries, query_vectors @ vectors.T):
        ranked = sorted(range(len(ids)), key=lambda i: (-float(scores[i]), ids[i]))
        top = [ids[i] for i in ranked]
        relevant = set(query["relevant"])
        ideal = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), 10)))
        dcg = sum(1 / math.log2(i + 2) for i, pid in enumerate(top[:10]) if pid in relevant)
        rows.append(dcg / ideal if ideal else 0.0)
        rankings.append(top)
    return float(np.mean(rows)), rankings


def embeddings(args):
    root, saved = args.benchmark, args.results
    documents = read_jsonl(root / "data/passages.jsonl")
    all_queries = read_jsonl(root / "data/queries.jsonl")
    results = {}
    for index, (model, benchmark_model) in enumerate(
        (("e5-small", "multilingual-e5-small"), ("user-bge-m3", "user-bge-m3-onnx"))
    ):
        base = f"{args.host}:{args.base_port + index}"
        docs_gpu = embed(base, model, [p["text"] for p in documents], "passage")
        results[model] = {}
        for split in ("tune", "holdout"):
            queries = [q for q in all_queries if q["split"] == split]
            query_gpu = embed(base, model, [q["query"] for q in queries], "query")
            folder = saved / f"quality-{split}"
            prefix = folder / f"{benchmark_model}.{split}"
            if not Path(str(prefix) + ".summary.json").is_file():
                # E5 tune quality was recovered with its successful CPU run.
                folder = saved / f"perf-{split}"
                prefix = folder / f"{benchmark_model}.{split}"
            summary = json.loads(Path(str(prefix) + ".summary.json").read_text())
            for field, source in (
                ("corpus_sha256", root / "data/passages.jsonl"),
                ("queries_sha256", root / "data/queries.jsonl"),
            ):
                assert summary[field] == hashlib.sha256(source.read_bytes()).hexdigest(), (
                    "benchmark inputs changed"
                )
            docs_cpu = np.load(str(prefix) + ".docs.npy")
            query_cpu = np.load(str(prefix) + ".queries.npy")
            assert docs_cpu.shape == docs_gpu.shape and query_cpu.shape == query_gpu.shape

            def cosine(a, b):
                return np.sum(a * b, axis=1) / (
                    np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
                )

            min_cosine = float(
                min(cosine(docs_gpu, docs_cpu).min(), cosine(query_gpu, query_cpu).min())
            )
            score_gpu, ranks_gpu = retrieval(documents, docs_gpu, queries, query_gpu)
            score_cpu, ranks_cpu = retrieval(documents, docs_cpu, queries, query_cpu)
            record = {
                "queries": len(queries),
                "min_cosine": min_cosine,
                "reference_ndcg10": score_cpu,
                "service_ndcg10": score_gpu,
                "delta_pp": 100 * (score_gpu - score_cpu),
                "top1_changes": sum(a[0] != b[0] for a, b in zip(ranks_gpu, ranks_cpu)),
            }
            assert min_cosine >= 0.99999, record
            assert score_gpu - score_cpu >= -args.max_degradation, record
            if model == "user-bge-m3":
                int8_prefix = folder / f"user-bge-m3-onnx-int8.{split}"
                int8_score, _ = retrieval(
                    documents,
                    np.load(str(int8_prefix) + ".docs.npy"),
                    queries,
                    np.load(str(int8_prefix) + ".queries.npy"),
                )
                record.update(
                    int8_ndcg10=int8_score, delta_to_int8_pp=100 * (score_gpu - int8_score)
                )
                assert score_gpu - int8_score >= -args.max_degradation, record
            results[model][split] = record
    return results


def ocr(args):
    from rapidfuzz.distance import Levenshtein

    def norm(text):
        return " ".join(unicodedata.normalize("NFKC", text).casefold().split())

    report = {}
    for name in ("corpus", "public"):
        root = args.ocr_root / name
        results = args.ocr_root / ("results" if name == "corpus" else "results-public")
        pages = read_jsonl(root / "manifest.jsonl")
        report[name] = {}
        for split in ("tune", "holdout"):
            selected = [p for p in pages if p["split"] == split]
            source = results / f"perf-{split}/rapid-v5-mobile.predictions.json"
            reference = {p["id"]: p["lines"] for p in json.loads(source.read_text())}
            gpu_errors = cpu_errors = chars = exact = 0
            for start in range(0, len(selected), 4):
                chunk = selected[start : start + 4]
                images = [
                    base64.b64encode((root / p["image"]).read_bytes()).decode() for p in chunk
                ]
                actual = request(
                    f"{args.host}:{args.base_port + 2}",
                    "/rapid-v5-mobile/ocr/batch",
                    {"images": images},
                )["pages"]
                for page, out in zip(chunk, actual):
                    expected_text = norm("\n".join(line["text"] for line in reference[page["id"]]))
                    actual_text = norm(out["text"])
                    truth = norm(page["text"])
                    chars += len(truth)
                    cpu_errors += Levenshtein.distance(truth, expected_text)
                    gpu_errors += Levenshtein.distance(truth, actual_text)
                    exact += actual_text == expected_text
            record = {
                "pages": len(selected),
                "exact_normalized_text": exact,
                "cpu_cer": cpu_errors / chars,
                "gpu_cer": gpu_errors / chars,
                "delta_pp": 100 * (gpu_errors - cpu_errors) / chars,
            }
            assert record["gpu_cer"] - record["cpu_cer"] <= args.max_degradation, record
            report[name][split] = record
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="http://127.0.0.1")
    p.add_argument("--base-port", type=int, default=18101)
    p.add_argument("--benchmark", type=Path)
    p.add_argument("--results", type=Path)
    p.add_argument("--ocr-root", type=Path)
    p.add_argument("--max-degradation", type=float, default=0.005)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    report = {"status": "passed", "max_degradation_pp": 100 * a.max_degradation}
    if a.benchmark:
        a.results = a.results or a.benchmark / "results-v3"
        report["embeddings"] = embeddings(a)
    if a.ocr_root:
        report["ocr"] = ocr(a)
    assert a.benchmark or a.ocr_root, "specify a benchmark"
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
