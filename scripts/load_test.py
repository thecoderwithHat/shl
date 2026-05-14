"""Run a randomized load test against the /chat endpoint.

Usage:
  python scripts/load_test.py --count 200 --concurrency 20

This script sends many randomized requests and prints a summary of latency and status.
"""
from __future__ import annotations
import argparse
import asyncio
import httpx
import random
import time
import json

BASE = "http://127.0.0.1:8000"

TEMPLATES = [
    "We're screening {n} {level} {role} candidates for {goal}.",
    "Hiring {n} {role} for {goal}. Need spoken language screening in {language}.",
    "Need assessments for {role} at {level} level; focus on {goal}.",
    "Looking for a shortlist for {role} - {goal} - {language} speakers.",
]

ROLES = [
    "contact centre agent",
    "customer service representative",
    "software engineer",
    "graduate trainee",
    "safety officer",
    "financial analyst",
]

LEVELS = ["entry-level", "mid", "senior", "manager"]
GOALS = ["screening", "selection", "development", "benchmarking"]
LANGUAGES = ["English", "Spanish", "French", "German", "Portuguese"]


def make_message():
    tmpl = random.choice(TEMPLATES)
    n = random.choice([10, 50, 100, 200, 500])
    role = random.choice(ROLES)
    level = random.choice(LEVELS)
    goal = random.choice(GOALS)
    language = random.choice(LANGUAGES)
    return tmpl.format(n=n, role=role, level=level, goal=goal, language=language)


async def worker(name: int, client: httpx.AsyncClient, q: asyncio.Queue, results: list):
    while True:
        item = await q.get()
        if item is None:
            q.task_done()
            break
        payload = {"messages": [{"role": "user", "content": item}]}
        start = time.perf_counter()
        try:
            r = await client.post(f"{BASE}/chat", json=payload, timeout=30.0)
            elapsed = time.perf_counter() - start
            try:
                data = r.json()
            except Exception:
                data = None
            results.append((r.status_code, elapsed, data))
        except Exception as e:
            elapsed = time.perf_counter() - start
            results.append((None, elapsed, str(e)))
        q.task_done()


async def run(count: int, concurrency: int, seed: int | None):
    random.seed(seed)
    q: asyncio.Queue = asyncio.Queue()
    results: list = []
    for _ in range(count):
        q.put_nowait(make_message())
    for _ in range(concurrency):
        q.put_nowait(None)

    async with httpx.AsyncClient() as client:
        workers = [asyncio.create_task(worker(i, client, q, results)) for i in range(concurrency)]
        start = time.perf_counter()
        await q.join()
        total = time.perf_counter() - start
        for w in workers:
            w.cancel()

    # Summarize
    statuses = {}
    latencies = [r[1] for r in results if r[1] is not None]
    success_count = sum(1 for r in results if r[0] == 200)
    for st, _, _ in results:
        statuses[st] = statuses.get(st, 0) + 1

    print(f"Total requests: {len(results)}")
    print(f"Success (200): {success_count}")
    print("Statuses:")
    # Normalize keys for display (None -> ERROR)
    items = [("ERROR" if k is None else str(k), v) for k, v in statuses.items()]
    for k, v in sorted(items, key=lambda kv: kv[0]):
        print(f"  {k}: {v}")
    if latencies:
        latencies_ms = [l * 1000 for l in latencies]
        print(f"Total elapsed: {total:.2f}s")
        print(f"Avg latency: {sum(latencies_ms)/len(latencies_ms):.1f} ms")
        print(f"P50: {sorted(latencies_ms)[len(latencies_ms)//2]:.1f} ms")
        print(f"P95: {sorted(latencies_ms)[max(0,int(len(latencies_ms)*0.95)-1)]:.1f} ms")

    # print a few sample replies
    print("\nSample replies:")
    for i, (_, _, data) in enumerate(results[:5]):
        print(f"--- sample {i+1} ---")
        try:
            print(json.dumps(data, indent=2))
        except Exception:
            print(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    asyncio.run(run(args.count, args.concurrency, args.seed))
