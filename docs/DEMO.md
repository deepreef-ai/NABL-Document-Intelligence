# Demo runbook

Everything needed to show the agentic extraction workflow working, including
what to do if something goes wrong mid-demo.

---

## Before you start (2 minutes)

**1. Verify the integration is intact.** Files that predate the graph work have
reverted repeatedly during development — losing the router registration, the
route, or the pipeline branch. One command tells you:

```bash
cd backend
.venv/Scripts/python.exe scripts/apply_integration.py --check
```

`integration intact` means you are good. If it lists anything missing, drop the
`--check` and it re-applies only what is gone, then restart both servers.

**2. Seed the demo runs** (already done once; safe to repeat):

```bash
.venv/Scripts/python.exe scripts/seed_demo_runs.py
```

Three completed runs appear, and they need **no LLM quota** — the model replies
are scripted, everything else is real. This matters: Gemini's free tier is a
handful of calls per day, and a live call failing mid-demo is the one thing you
cannot recover from gracefully.

**3. Start both servers** in separate terminals:

```bash
# terminal 1
cd backend
.venv/Scripts/python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8080

# terminal 2
cd frontend
VITE_API_BASE=http://localhost:8080 npx vite --port 5173
```

**4. Confirm before anyone is watching:**

```bash
curl http://localhost:8080/graph/health        # {"status":"ok", ...}
```

Then open **http://localhost:5173/extraction** and check the RECENT list shows
three runs. If it is empty, the UI cannot reach the API — see Troubleshooting.

---

## The demo, in the order that tells the story

### 1. Open with the clean run — "it reads the document properly"

Click **clean_milk_report.pdf** (VALIDATED).

Point at the green banner: *every page read, every value backed by the
document*. The six counters and the five quality checks are all derived from
the run, not decoration.

Then the main point — **the fields are grouped the way the document is
organised**:

```
▸ Header and Sample Information     page 1    5/5 verified
      Report No     LR-2024-0195     p1  [evidenced]  95%
      Sample Name   Cow Milk         p1  [evidenced]  95%
      ...
▸ Results                           page 2    5/5 verified
      Fat           4.2 %            p2  [evidenced]  95%
      Protein       3.4 g/100ml      p2  [evidenced]  95%
      ...
```

Those headings were named by the model's structure analysis of the actual
document. They are not a fixed list.

**Click any row.** It expands to the verbatim quote the value came from. Say
plainly: *that quote was checked back against the page text — if it were not
there, the value would be withheld.* This is the part that distinguishes the
system from "we asked a model and hoped".

### 2. Switch to the Form tab — "it fills the target form"

Five target fields. Four filled from verified values, `analysis.moisture`
reported **not found** rather than guessed. Each row names the source field and
the page, and the mapping reason is in the tooltip.

### 3. Open the conflicting run — "it does not guess"

Click **conflicting_report.pdf** (CONFLICTED), then the **Review** tab.

This document states `Report No: LR-2024-0311` on page 1 and `LR-2024-0312` in
an amended summary on page 3, and Fat as both `6.1 %` and `6.4 %`. The system
found both, kept both, and refused to choose:

> *the amended summary states a different figure and the document does not say
> which supersedes the other*

Note `✕ conflicts resolved` in red in the quality row — the run is honest about
being unresolved. Selecting a candidate and applying it resumes the workflow
from its checkpoint.

### 4. Open the third run — "it catches its own mistakes"

**unsupported_read.pdf** (MANUAL REVIEW REQUIRED). One extracted value cites a
quote that does not appear in the document. The evidence check caught it, the
value was withheld from the form, and it is listed for review rather than
deleted.

### 5. Audit tab — "everything is traceable"

Every node execution, in order, with timings and warnings. Useful if anyone
asks "how do you know it read page 2".

### 6. Optional: a live run

Only if quota allows. Upload any PDF and watch the progress panel step through
the named stages. **Have a fallback ready** — if it fails, that is the
provider, and the three seeded runs still demo everything.

---

## Questions you will get

**"What model is this?"** Nova is configured as primary; it is currently
blocked at the AWS account level (`ValidationException: Operation not allowed`
— an entitlement issue, not a code one), so runs use Gemini. The provider chain
is ordered, so it switches the moment access is granted.

**"How accurate is it?"** No accuracy figure exists for this pipeline yet.
Nothing has been scored against ground truth. Say that plainly rather than
quoting the old benchmark, which measured a different, simpler code path.

**"Will it handle our 300-page forms?"** Designed for it — chunking, bounded
retries, resumable checkpoints. Tested only up to 3 pages. Be straight about
this.

**"Can we deploy it?"** Not yet. No authentication on any endpoint, single-node
only, and no working production model.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| RECENT list empty | UI cannot reach the API | Check `curl localhost:8080/graph/health`; confirm `VITE_API_BASE`; check the browser console for a CORS error |
| `/extraction` shows the wizard | `App.tsx` reverted, route gone | `python scripts/apply_integration.py`, restart Vite |
| 404 on `/graph/*` | `main.py` reverted, router unregistered | `python scripts/apply_integration.py`, restart uvicorn |
| Backend will not start | `pipeline.py` reverted to a conflicted state | `python scripts/apply_integration.py` |
| Live upload fails | Gemini quota or Nova entitlement | Fall back to the seeded runs |
| CORS error in console | UI on a host the API does not allow | Use `localhost:5173`, or add the origin to `cors_origins` |

**If something breaks and you cannot fix it in 30 seconds, click a seeded run.**
They are stored results and need neither the LLM nor the graph to display.

---

## What is real, and what is scripted

Worth being precise if anyone technical is watching.

**Real in the seeded runs:** the PDFs, PyMuPDF text extraction, chunking,
evidence validation against page text, conflict detection, field normalisation,
form mapping rules, form-filling rules, quality control, the final decision, the
audit trail, and every number on screen.

**Scripted:** only the model's replies, so the runs are reproducible and cost
nothing.

**Fully live** (real Gemini, no scripting) has also been verified end to end —
`VALIDATED`, 8/8 fields evidenced, 5 model calls — it is just quota-limited.
