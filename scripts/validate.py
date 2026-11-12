#!/usr/bin/env python3
"""Mechanical checks (no LLM).
  python scripts/validate.py ledger   -> checks facts/ledger/*.jsonl, merges valid facts into facts/facts.jsonl
  python scripts/validate.py doc      -> checks knowledge.md citations against facts/facts.jsonl + decisions.md
Exit code 1 if any error."""
import json, re, sys
from pathlib import Path

ROOT = Path(".").resolve()
FACTS = ROOT / "facts"
REQ = ["id", "claim", "value", "source", "locator", "quote", "type"]
TYPES = {"extracted", "computed", "unknown"}
ID_RE = r"[A-Za-z0-9][A-Za-z0-9_.-]*-\d{2,}"

def norm(s):
    return re.sub(r"\s+", " ", str(s)).strip()

def cell_section(text, n):
    m = re.search(rf"^## \[cell {n}\][^\n]*\n(.*?)(?=^## \[cell \d+\]|\Z)", text, re.S | re.M)
    return m.group(1) if m else None

def check_ledger():
    errors, warns, facts, bad = [], [], {}, set()
    files = sorted((FACTS / "ledger").glob("*.jsonl"))
    if not files:
        print("ERROR: no files in facts/ledger/"); return 1
    for f in files:
        for ln, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            where = f"{f.name}:{ln}"
            try:
                fact = json.loads(line)
            except Exception as e:
                errors.append(f"{where}: invalid JSON ({e})"); continue
            miss = [k for k in REQ if k not in fact]
            if miss:
                errors.append(f"{where}: missing fields {miss}"); continue
            fid = fact["id"]
            if not re.fullmatch(ID_RE, str(fid)):
                errors.append(f"{where}: bad id '{fid}' (need SLUG-NNN)"); bad.add(fid)
            if fid in facts:
                errors.append(f"{where}: duplicate id {fid}"); bad.add(fid)
            facts[fid] = fact
            t = fact["type"]
            if t not in TYPES:
                errors.append(f"{where} {fid}: type must be one of {sorted(TYPES)}"); bad.add(fid); continue
            if t == "unknown":
                if fact["value"] != "UNKNOWN":
                    errors.append(f"{where} {fid}: unknown facts must have value 'UNKNOWN'"); bad.add(fid)
                continue
            if t == "computed":
                continue  # inputs checked below
            src = ROOT / str(fact["source"])
            if not src.is_file():
                errors.append(f"{where} {fid}: source file not found: {fact['source']}"); bad.add(fid); continue
            quote = norm(fact["quote"])
            if len(quote) < 3:
                errors.append(f"{where} {fid}: quote missing/too short"); bad.add(fid); continue
            text = src.read_text(encoding="utf-8", errors="replace")
            hay = text
            m = re.search(r"cell (\d+)", str(fact["locator"]))
            if m and str(fact["source"]).startswith("facts/nb/"):
                sec = cell_section(text, m.group(1))
                if sec is None:
                    errors.append(f"{where} {fid}: cell {m.group(1)} not in {fact['source']}"); bad.add(fid); continue
                hay = sec
            if quote not in norm(hay):
                errors.append(f"{where} {fid}: quote NOT FOUND verbatim in {fact['source']} ({fact['locator']})"); bad.add(fid)
            elif norm(fact["value"]) not in quote and fact["value"] != "UNKNOWN":
                warns.append(f"{where} {fid}: value '{fact['value']}' not literally inside quote")
    for fid, fact in facts.items():
        if fact["type"] == "computed":
            ins = fact.get("inputs")
            if not ins or not fact.get("method") or any(i not in facts for i in ins):
                errors.append(f"{fid}: computed facts need 'method' and 'inputs' naming existing fact ids"); bad.add(fid)
    good = [f for i, f in facts.items() if i not in bad]
    (FACTS / "facts.jsonl").write_text("\n".join(json.dumps(f, ensure_ascii=False) for f in good) + "\n", encoding="utf-8")
    for w in warns: print("WARN ", w)
    for e in errors: print("ERROR", e)
    print(f"\n{len(good)} valid facts written to facts/facts.jsonl; {len(bad)} rejected; {len(warns)} warnings")
    return 1 if errors else 0

def decision_ids():
    p = ROOT / "decisions.md"
    return set(re.findall(r"\b(D-\d+)\b", p.read_text(encoding="utf-8"))) if p.exists() else set()

def check_doc():
    fp, kp = FACTS / "facts.jsonl", ROOT / "knowledge.md"
    if not kp.exists() or not fp.exists():
        print("ERROR: need knowledge.md and facts/facts.jsonl"); return 1
    facts = {}
    for l in fp.read_text(encoding="utf-8").splitlines():
        if l.strip():
            d = json.loads(l); facts[d["id"]] = d
    dec = decision_ids()
    errors, warns, in_open, in_fence, checked = [], [], False, False, 0
    for ln, line in enumerate(kp.read_text(encoding="utf-8").splitlines(), 1):
        s = line.strip()
        if s.startswith("```"):
            in_fence = not in_fence; continue
        if in_fence or not s:
            continue
        if s.startswith("#"):
            in_open = "open question" in s.lower(); continue
        if re.fullmatch(r"[|\s:-]+", s) or s.startswith("<!--"):
            continue
        ids = []
        for grp in re.findall(r"\[([^\]]+)\]", s):
            parts = [x.strip() for x in grp.split(",")]
            if all(re.fullmatch(ID_RE, x) for x in parts):
                ids += parts
        if in_open:
            continue
        checked += 1
        if not ids:
            errors.append(f"line {ln}: UNCITED: {s[:90]}"); continue
        unknown = [i for i in ids if i not in facts and i not in dec]
        if unknown:
            errors.append(f"line {ln}: cites nonexistent id(s) {unknown}"); continue
        corpus = " ".join(f"{facts[i]['claim']} {facts[i]['value']} {facts[i]['quote']}" for i in ids if i in facts)
        body = re.sub(r"\[[^\]]*\]", " ", s)
        body = re.sub(r"^\s*(\d+\.|[-*])\s+", "", body)
        for tok in re.findall(r"(?<![\w.])\d+(?:\.\d+)?", body):
            if tok not in corpus:
                warns.append(f"line {ln}: number {tok} not found in cited facts")
    for w in warns: print("WARN ", w)
    for e in errors: print("ERROR", e)
    print(f"\n{checked} content lines checked; {len(errors)} errors; {len(warns)} warnings")
    return 1 if errors else 0

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    sys.exit({"ledger": check_ledger, "doc": check_doc}.get(mode, lambda: print(__doc__) or 2)())
