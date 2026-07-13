"""
Script di valutazione sul golden dataset.
Esegue la pipeline RAG per ogni query, verifica fonti e contenuto, stampa report.
"""

import json
import re
import sys
from pathlib import Path

from rag import NO_INFO_MESSAGE, ask, index_documents

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
REJECTION_PREFIX = NO_INFO_MESSAGE.split(".")[0]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def _check_answer_contains(answer: str, expected_strings: list[str]) -> tuple[bool, list[str]]:
    normalized_answer = _normalize(answer)
    missing = [s for s in expected_strings if _normalize(s) not in normalized_answer]
    return len(missing) == 0, missing


def _check_source(sources: list[str], answer: str, expected_source: str | None) -> bool:
    if expected_source is None:
        return REJECTION_PREFIX in answer
    return expected_source in sources or expected_source in answer


def run_evaluation() -> int:
    dataset = json.loads(GOLDEN_DATASET_PATH.read_text(encoding="utf-8"))

    print("Indicizzazione documenti (reindex forzato)...")
    chunk_count = index_documents(force_reindex=True)
    print(f"Chunk indicizzati: {chunk_count}\n")

    results: list[dict] = []

    for item in dataset:
        query = item["query"]
        expected_source = item["expected_source"]
        expected_contains = item["expected_answer_contains"]

        print(f"[{item['id']}] {query}")

        try:
            response = ask(query)
            answer = response["answer"]
            sources = response["sources"]
        except Exception as exc:
            print(f"  ERRORE: {exc}")
            results.append(
                {
                    "id": item["id"],
                    "query": query,
                    "passed": False,
                    "sources": [],
                    "content_ok": False,
                    "source_ok": False,
                    "reason": f"Eccezione: {exc}",
                }
            )
            print()
            continue

        content_ok, missing = _check_answer_contains(answer, expected_contains)
        source_ok = _check_source(sources, answer, expected_source)
        passed = content_ok and source_ok

        reason_parts: list[str] = []
        if not content_ok:
            reason_parts.append(f"stringhe mancanti: {missing}")
        if not source_ok:
            if expected_source is None:
                reason_parts.append("risposta non ha rifiutato correttamente")
            else:
                reason_parts.append(
                    f"fonte attesa '{expected_source}' non trovata in {sources}"
                )

        status = "PASS" if passed else "FAIL"
        print(f"  {status} | fonti={sources}")
        if not passed:
            print(f"  Motivo: {'; '.join(reason_parts)}")
            print(f"  Risposta: {answer[:250]}{'...' if len(answer) > 250 else ''}")

        results.append(
            {
                "id": item["id"],
                "query": query,
                "passed": passed,
                "sources": sources,
                "content_ok": content_ok,
                "source_ok": source_ok,
                "reason": "; ".join(reason_parts),
            }
        )
        print()

    total = len(results)
    passed_count = sum(1 for r in results if r["passed"])
    accuracy = (passed_count / total * 100) if total else 0.0

    print("=" * 60)
    print(f"ACCURATEZZA: {passed_count}/{total} ({accuracy:.1f}%)")
    print("=" * 60)

    print("\nRiepilogo per test:")
    print(f"{'ID':<6} {'Fonte':<8} {'Contenuto':<10} {'Esito':<6}")
    print("-" * 36)
    for r in results:
        print(
            f"{r['id']:<6} "
            f"{'OK' if r.get('source_ok') else 'FAIL':<8} "
            f"{'OK' if r.get('content_ok') else 'FAIL':<10} "
            f"{'PASS' if r['passed'] else 'FAIL':<6}"
        )

    failures = [r for r in results if not r["passed"]]
    if failures:
        print("\nDettaglio fallimenti:")
        for fail in failures:
            print(f"  [{fail['id']}] {fail['reason']}")
            print(f"         Domanda: {fail['query']}")

    return 0 if passed_count == total else 1


if __name__ == "__main__":
    sys.exit(run_evaluation())
