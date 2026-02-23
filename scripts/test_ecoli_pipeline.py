"""E. coli discovery pipeline test with LLM speed measurement."""

from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, "src")

from virtualfermlab.discovery.paper_search import fetch_full_text, search_pubmed
from virtualfermlab.discovery.llm_extraction import LLMClient
import requests


def main() -> None:
    base_url = "http://erc-hpc-comp245:8000/v1"
    client = LLMClient(base_url=base_url)

    if not client.is_available():
        print(f"vLLM not available at {base_url}")
        return

    print("=" * 70)
    print("  E. coli Fermentation Kinetics — Discovery Pipeline Test")
    print("=" * 70)

    # --- Step 1: Search PubMed ---
    print("\n[1] Searching PubMed for E. coli fermentation kinetics...")
    papers = search_pubmed(
        '"Escherichia coli" fermentation kinetics Monod glucose',
        max_results=10,
    )
    print(f"    Found {len(papers)} papers\n")

    if not papers:
        print("No papers found.")
        return

    # --- Step 2: Full-text fetch ---
    print("[2] Fetching full-text from PMC...")
    ft_count = 0
    for p in papers:
        ft = fetch_full_text(p)
        if ft:
            p["full_text"] = ft
            ft_count += 1
            print(f"    OK  {p['pmid']:>10s} | {len(ft):>5d} chars | {p['title'][:60]}")
        else:
            print(f"    --  {p.get('pmid','?'):>10s} | no PMC   | {p['title'][:60]}")
    print(f"\n    Full-text available: {ft_count}/{len(papers)}\n")

    # --- Step 3: LLM extraction with speed measurement ---
    print("[3] LLM extraction (with speed measurement)...")
    print("-" * 70)

    all_params = []
    speed_records = []

    for i, paper in enumerate(papers):
        if not paper.get("abstract") and not paper.get("full_text"):
            continue

        mode = "full-text" if paper.get("full_text") else "abstract"
        title_short = paper["title"][:55]

        # Measure LLM call time
        t0 = time.perf_counter()

        # Make raw API call to also capture token counts
        from virtualfermlab.discovery.prompts import (
            EXTRACTION_PROMPT_FULLTEXT,
            EXTRACTION_PROMPT_ABSTRACT,
            EXTRACTION_SYSTEM_PROMPT,
        )

        if paper.get("full_text"):
            prompt = EXTRACTION_PROMPT_FULLTEXT.format(
                title=paper.get("title", ""),
                full_text=paper["full_text"],
            )
        else:
            prompt = EXTRACTION_PROMPT_ABSTRACT.format(
                title=paper.get("title", ""),
                abstract=paper.get("abstract", ""),
            )

        payload = {
            "model": client.model,
            "messages": [
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 2048,
        }

        try:
            resp = requests.post(
                f"{base_url}/chat/completions",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            result = resp.json()
            elapsed = time.perf_counter() - t0

            # Extract token usage
            usage = result.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)

            # Speed calculations
            output_tok_per_sec = completion_tokens / elapsed if elapsed > 0 else 0
            total_tok_per_sec = total_tokens / elapsed if elapsed > 0 else 0

            speed_records.append({
                "mode": mode,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "elapsed_s": elapsed,
                "output_tok_per_sec": output_tok_per_sec,
            })

            # Parse the extraction result
            content = result["choices"][0]["message"]["content"]
            params = client.extract_params(paper)
            param_count = len(params)
            all_params.extend(params)

            print(f"  [{i+1:2d}] {mode:>9s} | {elapsed:5.1f}s | "
                  f"in:{prompt_tokens:>5d} out:{completion_tokens:>4d} tok | "
                  f"{output_tok_per_sec:5.1f} tok/s | "
                  f"{param_count} params | {title_short}")

        except Exception as e:
            elapsed = time.perf_counter() - t0
            print(f"  [{i+1:2d}] {mode:>9s} | ERROR ({elapsed:.1f}s): {e}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    print(f"\n  Papers searched  : {len(papers)}")
    print(f"  Full-text fetched: {ft_count}")
    print(f"  Total params     : {len(all_params)}")

    if all_params:
        print(f"\n  Extracted parameters:")
        for p in all_params:
            print(f"    {p['strain_name'] or '(unnamed)':25s} | "
                  f"{p['name']:8s} = {p['value']:<8g} {p.get('unit',''):6s} | "
                  f"substrate: {p.get('substrate', '-')}")

    if speed_records:
        print(f"\n  LLM Speed Statistics:")
        print(f"  {'Mode':>10s}  {'Prompt':>7s}  {'Output':>7s}  {'Time':>6s}  {'Output tok/s':>12s}")
        print(f"  {'-'*10}  {'-'*7}  {'-'*7}  {'-'*6}  {'-'*12}")
        for r in speed_records:
            print(f"  {r['mode']:>10s}  {r['prompt_tokens']:>7d}  {r['completion_tokens']:>7d}  "
                  f"{r['elapsed_s']:>5.1f}s  {r['output_tok_per_sec']:>10.1f}")

        avg_out_speed = sum(r["output_tok_per_sec"] for r in speed_records) / len(speed_records)
        avg_elapsed = sum(r["elapsed_s"] for r in speed_records) / len(speed_records)
        total_prompt = sum(r["prompt_tokens"] for r in speed_records)
        total_output = sum(r["completion_tokens"] for r in speed_records)
        total_time = sum(r["elapsed_s"] for r in speed_records)

        # Rough words estimate (~0.75 words per token for English)
        words_per_sec = avg_out_speed * 0.75

        print(f"\n  Average output speed : {avg_out_speed:.1f} tokens/sec (~{words_per_sec:.1f} words/sec)")
        print(f"  Average latency      : {avg_elapsed:.1f} sec/paper")
        print(f"  Total tokens         : {total_prompt} prompt + {total_output} output")
        print(f"  Total wall time      : {total_time:.1f} sec")


if __name__ == "__main__":
    main()
