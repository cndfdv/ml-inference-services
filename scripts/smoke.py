#!/usr/bin/env python3
"""Real HTTP, ordered batches and concurrent clients. No model imports."""

import argparse
import base64
import concurrent.futures
import io
import json
import urllib.error
import urllib.request


def request(base, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        base + path, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=600) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://127.0.0.1")
    parser.add_argument("--base-port", type=int, default=18101)
    args = parser.parse_args()
    results = {}
    for offset, name in enumerate(
        ("e5-small", "user-bge-m3", "rapid-v5-mobile", "whisper-large-v3")
    ):
        base = f"{args.host}:{args.base_port + offset}"
        assert request(base, "/readyz")["status"] == "ready"
        results[name] = request(base, "/v1/models")["data"][0]
        if offset >= 2:
            continue
        texts = ["Проверка поиска документов", "Заявка номер 884", "Срок хранения договоров"]
        output = request(base, f"/{name}/embed", {"texts": texts, "role": "query"})
        vectors = output["embeddings"]
        assert len(vectors) == len(texts)
        dimension = 384 if name == "e5-small" else 1024
        assert all(len(v) == dimension and abs(sum(x * x for x in v) - 1) < 1e-5 for v in vectors)

        def one(index, base=base, name=name, texts=texts):
            response = request(
                base, f"/{name}/embed", {"texts": [texts[index % 3]], "role": "query"}
            )
            return index, response["embeddings"][0]

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            for index, vector in pool.map(one, range(16)):
                assert max(abs(a - b) for a, b in zip(vector, vectors[index % 3])) < 1e-5
        results[name]["concurrent_requests"] = 16
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (400, 200), "white").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    output = request(
        f"{args.host}:{args.base_port + 2}",
        "/rapid-v5-mobile/ocr/batch",
        {"images": [encoded, encoded]},
    )
    assert len(output["pages"]) == 2 and all(p["text"] == "" for p in output["pages"])
    print(json.dumps({"status": "passed", "services": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
