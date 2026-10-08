#!/usr/bin/env python3
"""Confirm multiple loaded CPU workers through real HTTP connections."""

import argparse
import concurrent.futures
import json
import time

from smoke import request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:18201")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    deadline = time.monotonic() + 180
    seen = set()

    def sample(_):
        info = request(args.base, "/v1/models")["data"][0]
        assert info["metadata"]["device"] == "cpu"
        output = request(args.base, "/e5-small/embed", {"texts": ["Проверка двух воркеров"]})
        assert output["count"] == 1 and output["dim"] == 384
        return info["worker_pid"]

    while time.monotonic() < deadline and len(seen) < args.workers:
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                seen.update(pool.map(sample, range(32)))
        except OSError:
            time.sleep(1)
    assert len(seen) == args.workers, f"observed worker PIDs: {seen}"
    print(json.dumps({"status": "passed", "device": "cpu", "worker_pids": sorted(seen)}))


if __name__ == "__main__":
    main()
