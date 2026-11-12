#!/usr/bin/env python3
"""Stage 0a: turn heavy .ipynb files into small readable .md files with [cell N] anchors.
No LLM, stdlib only. Usage: python scripts/extract_notebooks.py [project_root]"""
import json, sys
from pathlib import Path

HEAD, TAIL = 1500, 1500  # keep start and end of long outputs (final results are usually at the end)

def text_of(x):
    return "".join(x) if isinstance(x, list) else (x or "")

def render_outputs(outputs):
    parts, skipped = [], 0
    for o in outputs:
        t = o.get("output_type")
        if t == "stream":
            parts.append(text_of(o.get("text")))
        elif t in ("execute_result", "display_data"):
            data = o.get("data", {})
            if "text/plain" in data:
                parts.append(text_of(data["text/plain"]))
            skipped += sum(1 for k in data if k != "text/plain")
        elif t == "error":
            parts.append(f"ERROR {o.get('ename')}: {o.get('evalue')}")
    txt = "\n".join(p.replace("\r\n", "\n").replace("\r", "\n") for p in parts).strip()
    if len(txt) > HEAD + TAIL:
        cut = len(txt) - HEAD - TAIL
        txt = txt[:HEAD] + f"\n[... {cut} chars truncated ...]\n" + txt[-TAIL:]
    return txt, skipped

def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    out_dir = root / "facts" / "nb"
    out_dir.mkdir(parents=True, exist_ok=True)
    for nb in sorted((root / "notebooks").glob("*.ipynb")):
        data = json.loads(nb.read_text(encoding="utf-8"))
        cells = data.get("cells", [])
        body, n_img, n_text_out, n_err = [], 0, 0, 0
        for i, c in enumerate(cells):
            kind = c.get("cell_type")
            src = text_of(c.get("source")).rstrip()
            body.append(f"## [cell {i}] {kind}")
            if kind == "code":
                body.append("```python\n" + src + "\n```")
                txt, skipped = render_outputs(c.get("outputs", []))
                n_img += skipped
                if txt:
                    n_text_out += 1
                    n_err += "ERROR " in txt
                    body.append("OUTPUT:\n```text\n" + txt + "\n```")
            else:
                body.append(src)
            body.append("")
        header = (f"# {nb.name}\n"
                  f"- cells: {len(cells)}\n- code cells with text output: {n_text_out}\n"
                  f"- non-text outputs omitted (images/html): {n_img}\n"
                  f"- cells whose output contains an error: {n_err}\n")
        (out_dir / (nb.stem + ".md")).write_text(header + "\n" + "\n".join(body), encoding="utf-8")
        print(f"{nb.name}: {len(cells)} cells, {n_text_out} text outputs, {n_img} images dropped")

if __name__ == "__main__":
    main()
