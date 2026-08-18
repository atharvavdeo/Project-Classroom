"""GBNF grammar validated against a live llama.cpp server.

This proves the mechanism PRD 9.2 depends on: that the grammar is syntactically
valid, that llama.cpp accepts it, and that a real language model generating
under it cannot emit anything the schema parser rejects.

It deliberately uses a tiny model. The claim being tested is a property of the
grammar and the sampler, not of Gemma's judgement -- a 135M model that has no
idea what an examination hall is will still be forced into valid output, and
that is exactly the point. If the constraint holds for a model with nothing
useful to say, it holds for a capable one.

Gemma's actual visual judgement is a separate question, tested separately once
the multimodal weights are present.

    tools/llamacpp/llama-server.exe -m models/tiny/SmolLM2-135M-Instruct-Q2_K.gguf \
        -c 2048 --port 8099
    python tests/test_grammar_live.py
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classroom import verify  # noqa: E402
from classroom.config import VerifyConfig  # noqa: E402

BASE = "http://127.0.0.1:8099"
BUDGET = VerifyConfig().max_tokens
ATTEMPTS = 8


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if condition else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    return bool(condition)


def server_up() -> bool:
    try:
        with urllib.request.urlopen(f"{BASE}/health", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def complete(prompt: str, grammar: str | None, n_predict: int = 200,
             temperature: float = 0.8) -> tuple[str, int]:
    payload = {"prompt": prompt, "n_predict": n_predict,
               "temperature": temperature, "cache_prompt": False}
    if grammar is not None:
        payload["grammar"] = grammar
    request = urllib.request.Request(
        f"{BASE}/completion", data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.loads(response.read())["content"], 200
    except urllib.error.HTTPError as exc:
        return exc.read().decode("utf-8", "replace"), exc.code


def main() -> int:
    if not server_up():
        print(f"SKIP: no llama-server at {BASE}")
        print("Start it with:\n  tools/llamacpp/llama-server.exe "
              "-m models/tiny/SmolLM2-135M-Instruct-Q2_K.gguf -c 2048 --port 8099")
        return 0

    ok = True
    prompt = ("Describe what is visible in an examination hall photograph. "
              "Respond in JSON.\n")

    print("llama.cpp accepts the grammar")
    text, status = complete(prompt, verify.GRAMMAR, n_predict=8)
    ok &= check("grammar compiles server-side", status == 200,
                f"HTTP {status}" + ("" if status == 200 else f" {text[:160]}"))
    if status != 200:
        print("\nFAILURES PRESENT")
        return 1

    print(f"\nconstrained generation over {ATTEMPTS} samples at temperature 1.0")
    parsed_ok = 0
    for i in range(ATTEMPTS):
        text, status = complete(prompt, verify.GRAMMAR, n_predict=BUDGET, temperature=1.0)
        if status != 200:
            ok &= check(f"  sample {i} returned 200", False, text[:120])
            continue
        try:
            v = verify.Verification.parse(text)
            v.validate()
            parsed_ok += 1
            preview = text.replace("\n", " ")[:88]
            print(f"  sample {i}: valid  obj={v.object_assessment:11s} "
                  f"quality={v.evidence_quality:12s} conf={v.confidence}")
            if i == 0:
                print(f"    raw: {preview}...")
        except Exception as exc:
            ok &= check(f"  sample {i} parsed", False,
                        f"{type(exc).__name__}: {text[:120]}")

    ok &= check("every constrained sample is schema-valid",
                parsed_ok == ATTEMPTS, f"{parsed_ok}/{ATTEMPTS}")

    print("\nbounded repetition prevents runaway generation")
    lengths = []
    for _ in range(4):
        text, status = complete(prompt, verify.GRAMMAR, n_predict=400, temperature=1.0)
        if status == 200:
            lengths.append(len(text))
    ok &= check("responses terminate well inside the budget",
                all(n < 2200 for n in lengths), f"max {max(lengths) if lengths else 0} chars")

    ok &= check("every constrained sample is schema-valid",
                parsed_ok == ATTEMPTS, f"{parsed_ok}/{ATTEMPTS}")

    print("\nenumerated fields cannot leave their vocabulary")
    # Regenerate and confirm no sample invents a value outside the enums.
    values = {"object": set(), "interaction": set(), "quality": set()}
    for _ in range(ATTEMPTS):
        text, status = complete(prompt, verify.GRAMMAR, n_predict=BUDGET, temperature=1.0)
        if status != 200:
            continue
        try:
            v = verify.Verification.parse(text)
        except Exception:
            continue
        values["object"].add(v.object_assessment)
        values["interaction"].add(v.interaction_assessment)
        values["quality"].add(v.evidence_quality)
    ok &= check("object values all in vocabulary",
                values["object"] <= set(verify.OBJECT_VALUES), str(values["object"]))
    ok &= check("interaction values all in vocabulary",
                values["interaction"] <= set(verify.INTERACTION_VALUES),
                str(values["interaction"]))
    ok &= check("quality values all in vocabulary",
                values["quality"] <= set(verify.QUALITY_VALUES), str(values["quality"]))

    print("\nwithout the grammar, the same model is not schema-valid")
    free, status = complete(prompt, None, n_predict=200, temperature=1.0)
    free_valid = False
    if status == 200:
        try:
            verify.Verification.parse(free)
            free_valid = True
        except Exception:
            free_valid = False
    ok &= check("unconstrained output fails the schema", not free_valid,
                f"raw: {free.replace(chr(10), ' ')[:80]}...")

    print("\na malformed grammar is rejected rather than ignored")
    broken, status = complete(prompt, 'root ::= "{" unterminated', n_predict=8)
    ok &= check("server rejects a broken grammar", status != 200, f"HTTP {status}")

    print("\nsanitisation applies to real generated output")
    text, status = complete(prompt, verify.GRAMMAR, n_predict=BUDGET, temperature=1.0)
    if status == 200:
        try:
            v = verify.Verification.parse(text)
            clean, removed = verify.sanitize(v)
            ok &= check("generated output survives sanitisation",
                        isinstance(clean, verify.Verification))
            ok &= check("no verdict language reaches the reviewer",
                        not verify.contains_verdict(clean.review_note)
                        and not any(verify.contains_verdict(o)
                                    for o in clean.supported_observations))
        except Exception as exc:
            ok &= check("generated output parsed", False, str(exc)[:100])

    print(f"\n{'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
