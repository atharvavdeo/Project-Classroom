<!-- code-review-graph MCP tools -->
## MCP Tools: code-review-graph

**This project has a knowledge graph. Start with the code-review-graph
MCP tools to narrow scope, then read the source.** The graph is cheaper than scanning files and
gives you structural context (callers, dependents, test coverage) that file search cannot.

### When to use graph tools FIRST

- **Exploring code**: `semantic_search_nodes_tool` or `query_graph_tool` instead of Grep
- **Understanding impact**: `get_impact_radius_tool` instead of manually tracing imports
- **Code review**: `detect_changes_tool` + `get_review_context_tool` instead of reading entire files
- **Finding relationships**: `query_graph_tool` with callers_of/callees_of/imports_of/tests_for
- **Architecture questions**: `get_architecture_overview_tool` + `list_communities_tool`

### Verify in the source

- Narrow scope with the graph, then read the source. Do not change code from graph output alone.
- For any non-trivial change, read the implementation and the relevant tests before concluding.
- Verify the exact source when touching behavior, database logic, migrations, retries, fallbacks,
  recovery, or compatibility code.
- When the graph and the source disagree, the source wins. The graph may be stale or may not
  model that relationship.
- An empty graph result can mean "not indexed" or "not statically visible", not "does not exist".

### Key Tools

| Tool | Use when |
| ------ | ---------- |
| `detect_changes_tool` | Reviewing code changes — gives risk-scored analysis |
| `get_review_context_tool` | Need source snippets for review — token-efficient |
| `get_impact_radius_tool` | Understanding blast radius of a change |
| `get_affected_flows_tool` | Finding which execution paths are impacted |
| `query_graph_tool` | Tracing callers, callees, imports, tests, dependencies |
| `semantic_search_nodes_tool` | Finding functions/classes by name or keyword |
| `get_architecture_overview_tool` | Understanding high-level codebase structure |
| `refactor_tool` | Planning renames, finding dead code |

### Workflow

1. The graph auto-updates on file changes (via hooks).
2. Use `detect_changes_tool` for code review.
3. Use `get_affected_flows_tool` to understand impact.
4. Use `query_graph_tool` pattern="tests_for" to check coverage.
<!-- /code-review-graph MCP tools -->

## Roboflow integrations

Two optional stages call hosted Roboflow endpoints. Read these before changing
either — both encode failures that are easy to reintroduce.

| File | What it is |
| ---- | ---------- |
| `pipeline/chit_detector.py` | `chit-paper-new/2`, a single-class paper detector |
| `pipeline/roboflow_workflow.py` | workflow `general-segmentation-api-6` — **SAM 3**, prompted with free text via the `classes` parameter |
| `tools/adjudicate_with_sam3.py` | uses SAM 3 as a referee over chit detections |

Rules for this code:

- **The API key comes from `ROBOFLOW_API_KEY`, never a literal.** Both clients
  raise if it is unset.
- **Do not add `inference-sdk`.** It declares `Requires-Python <3.13` and this
  project runs 3.13. The REST contract is one POST; `requests` is the
  established pattern here.
- **Parse against the workflow's declared output names** (`annotated_image`,
  `predictions`), and treat a 200 with the wrong shape as an error. An empty
  result must mean "the model found nothing", never "the call did not happen".
- **`class` values arrive with leading whitespace** (`" keyboard"`) because the
  prompt is split on commas without trimming. Always `.strip()`.
- **Never carry `rle_mask` or `points`** into pipeline state, and never log the
  base64 `annotated_image` — it is ~400 KB per call.
- **Keep concurrency at ~6.** A 16-worker burst lost 2,937 of 3,439 calls to
  rate limiting, and the failure counter is what caught it.
- These stages are **online**, which conflicts with PS 2's offline requirement.
  Say so wherever results from them are reported.
## The output contract (v1.0.0)

`docs/OUTPUT_CONTRACT.md` is frozen. Before changing anything that alters what
this system reports, read it — the dashboard and any future evaluation both
depend on these shapes.

| File | What it is |
| ---- | ---------- |
| `pipeline/reason_codes.py` | closed vocabulary, 36 codes; `get()` raises on anything invented |
| `pipeline/evidence_record.py` | one immutable record per proposal, rejections included |
| `pipeline/fusion.py` | the six-state policy machine |
| `tools/build_evidence_records.py` | converts a finished run into records + outcomes |
| `tests/test_output_contract.py` | 27 invariant tests |

Rules for this code:

- **Detectors propose; nothing judges.** `human_confirmed` and
  `human_dismissed` may only be written by a reviewer.
  `fusion.assert_machine_state()` enforces this.
- **`unsupported` means *not confirmed*, never *false* or *safe*.** SAM 3
  missed known true positives on this corpus. It maps to `SAM3_NOT_CONFIRMED`
  → `needs_better_view`.
- **A SAM 3 outage may never become a negative finding.** That is what
  `Sam3Response.responded` exists to separate from "found nothing".
- **COCO / D-FINE `cell phone` may not route anyone to review.** It is
  workstation context (`DFINE_OBJECT_CONTEXT`) and nothing more. The phone
  route runs through SAM 3 naming it (`SAM3_PHONE_NAMED`).
- **Records are never deleted, only annotated.** Suppression is a reason code
  written into the record, so precision and recall stay computable once
  ground truth exists.
- **Do not invent reason strings at a call site.** Add the code to
  `reason_codes.py` with a route and a guardrail.
- **Never present a count of proposals as accuracy or as a cheating rate.**
  No accuracy figure exists for this system until the evaluation pack in
  `OUTPUT_CONTRACT.md` §9 is built.

