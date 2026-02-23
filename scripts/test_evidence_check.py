"""Verify evidence + confidence for extracted parameters."""

from __future__ import annotations

import sys
import textwrap

sys.path.insert(0, "src")

from virtualfermlab.discovery.paper_search import fetch_full_text, search_pubmed
from virtualfermlab.discovery.llm_extraction import LLMClient, _evidence_contains_value


def main() -> None:
    base_url = "http://erc-hpc-comp245:8000/v1"
    client = LLMClient(base_url=base_url)

    if not client.is_available():
        print(f"vLLM not available at {base_url}")
        return

    print("=" * 80)
    print("  Evidence Verification Test")
    print("=" * 80)

    # --- Paper 1: Mycoprotein paper (known to have good kinetic data) ---
    print("\n[1] Fetching mycoprotein paper (PMID 39555020, PMC11565039)...")
    papers = search_pubmed(
        '"mycoprotein" "lignocellulosic" "parameter estimation"',
        max_results=5,
    )

    target = None
    for p in papers:
        if p.get("pmid") == "39555020":
            target = p
            break

    if not target:
        print("    Could not find target paper. Trying broader search...")
        papers = search_pubmed("39555020", max_results=3)
        for p in papers:
            if p.get("pmid") == "39555020":
                target = p
                break

    if not target:
        print("    FAILED: Target paper not found")
        return

    print(f"    Found: {target['title'][:70]}")

    # Fetch full text
    ft = fetch_full_text(target)
    if ft:
        target["full_text"] = ft
        print(f"    Full-text: {len(ft)} chars")
    else:
        print("    WARNING: No full-text available, using abstract only")

    # --- Extract with evidence ---
    print("\n[2] Extracting parameters with evidence...")
    params = client.extract_params(target)

    if not params:
        print("    No parameters extracted!")
        return

    print(f"    Extracted {len(params)} parameters\n")

    # --- Verify evidence ---
    print("[3] Evidence verification:")
    print("-" * 80)

    matched = 0
    mismatched = 0
    no_evidence = 0

    for i, p in enumerate(params):
        name = p["name"]
        value = p["value"]
        unit = p.get("unit", "")
        evidence = p.get("evidence")
        confidence = p.get("confidence", "?")
        substrate = p.get("substrate", "-")

        # Manual cross-check
        manual_check = _evidence_contains_value(evidence, value)

        if evidence is None:
            status = "NO EVIDENCE"
            no_evidence += 1
        elif manual_check:
            status = "MATCH"
            matched += 1
        else:
            status = "MISMATCH"
            mismatched += 1

        # Consistency check: does confidence agree with our manual check?
        confidence_ok = (
            (confidence == "A" and manual_check) or
            (confidence == "B" and not manual_check) or
            (confidence == "B" and evidence is None)
        )
        consistency = "OK" if confidence_ok else "INCONSISTENT"

        print(f"  [{i+1}] {name:10s} = {value:<10g} {unit:6s} | "
              f"substrate: {substrate or '-':10s} | "
              f"conf: {confidence} | {status}")

        if evidence:
            # Wrap long evidence text
            wrapped = textwrap.fill(evidence, width=72, initial_indent="      evidence: ",
                                    subsequent_indent="                ")
            print(wrapped)
        else:
            print("      evidence: (none)")

        if not confidence_ok:
            print(f"      WARNING: confidence={confidence} but manual_check={manual_check}")
        print()

    # --- Summary ---
    print("=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    total = len(params)
    print(f"  Total parameters : {total}")
    print(f"  Evidence MATCH   : {matched} ({matched/total*100:.0f}%) — value found in evidence, confidence=A")
    print(f"  Evidence MISMATCH: {mismatched} ({mismatched/total*100:.0f}%) — evidence provided but value not found, confidence=B")
    print(f"  No evidence      : {no_evidence} ({no_evidence/total*100:.0f}%) — LLM didn't provide evidence")

    if matched == total:
        print("\n  ALL parameters have matching evidence with confidence A!")
    elif mismatched > 0:
        print(f"\n  {mismatched} parameter(s) have evidence that doesn't contain the value.")
        print("  This may indicate LLM paraphrasing rather than quoting verbatim.")


if __name__ == "__main__":
    main()
