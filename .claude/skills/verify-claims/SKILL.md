---
name: verify-claims
description: Procedure and verdict definitions for auditing knowledge.md against the fact ledger and original sources. Read before verifying anything.
---
# Verifying claims

## Mindset
You are the last line of defense against hallucination. A "looks fine" verdict is a failure of your job unless you actually opened the evidence. Writers tend to paraphrase upward (adding reasons, superlatives, generalizations); catch that.

## Verdicts
- **PASS**: every assertion in the sentence is directly established by the cited fact(s), and (where you checked) by the original source.
- **FAIL**: the sentence contradicts the evidence (wrong number, wrong run, wrong metric, wrong direction of comparison).
- **UNSUPPORTED**: the sentence says more than the evidence establishes (causes, "because", "best", generalizations, rounding, implied meaning of a folder name, anything not in a cited fact).

## Checks per sentence
1. Every cited id exists. 2. Subject matches (right run, right shot count, right dataset split). 3. Number identical to the fact's value, same units, same metric name. 4. No extra assertions. 5. If it compares things, both sides cited and the comparison recomputed.

## Spot-checking sources
Open the original file named in the fact's `source`, find `quote` at `locator`, and read the surrounding lines. Ask: does the surrounding context change the meaning (e.g. the value is a default that gets overridden later, a commented-out line, an old cell that was re-run differently)? Check at least 25% of facts, and all facts behind numbers.

## Recomputing
Use Python with csv/pandas on facts/metrics/metrics_table.csv. Print the real numbers into the report. Watch for "higher is better" assumptions: confirm the metric direction (loss vs score) from the evidence, otherwise mark UNSUPPORTED.

## Coverage
Every `unknown` fact must show up in Open questions. Every CONFLICT note must show up in Anomalies. Every inventory flag must be mentioned.

## Report
Write facts/verification_report.md using the table format in your agent instructions. Finish with `OVERALL: PASS` or `OVERALL: FAIL`. Do not edit knowledge.md.
