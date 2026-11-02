# Evaluation report

## 1. Summary

I evaluated the submitted configuration (typed tools over PostgreSQL + handbook retrieval from Qdrant, `gemini-3.5-flash-lite`) on the 8 supplied sample questions and 24 questions I wrote myself (32 total, including 3 multi-turn questions that pass a `history` list).

| | Count | Share |
|---|---|---|
| Correct | 19 | 59% |
| Partly correct | 8 | 25% |
| Wrong | 5 | 16% |

- **Questions answered from the spreadsheets (price, promotion, service centres, fees): 13 correct, 4 partly correct, 0 wrong out of 17.** Date-sensitive rules (price-row effective dates, inclusive promotion end date, the 2,000,000 đ cap, temporary closures, the hidden 2025 price sheet) were handled correctly.
- **Multi-turn questions: 3 correct out of 3** (a follow-up to a clarifying question, a "what about Hà Nội?" follow-up, and a policy follow-up).
- **Questions that depend on the handbook or on the limits of the materials: 3 correct, 4 partly correct, 5 wrong out of 12.** This is the main weakness. Retrieval often returns unrelated clauses, and the handbook-over-spreadsheet precedence rule (clause 1.3) does not reliably reach the model.
- **Latency does not meet the target.** Server-side p50 is 5.7 s and p95 is 15.3 s (target ≤ 3 s / ≤ 7 s). Every request makes an extra validation-retry LLM call; without it, the same requests would have been about 4.1 s / 8.9 s (derived from step timings, not re-measured). The p95 is dominated by a few single LLM calls of 5–9 s on a rate-limited free-tier key.

## 2. Method

**Test set.** 32 questions: the 8 samples (S01–S08), 21 single-turn questions of my own (T01–T21) and 3 multi-turn questions (M1–M3). My own questions cover: price rows that change over time, promotion caps and inclusive end dates, a timestamp just after midnight, a discontinued model and the hidden 2025 price sheet, a province missing from the price table, handbook-versus-spreadsheet conflicts (30 vs 45 day activation window), commercial use, returns (online vs store, deadline arithmetic), service centres (Sunday hours, temporary closure), fee arithmetic, compensation and escalation, and questions the materials cannot answer (stock availability, out-of-scope products).

**Reference answers.** For each question I wrote a reference answer from the spreadsheets and handbook, including the sheet/row or clause it comes from and the specific facts that must appear. [Confirm before submitting: describe exactly how the references were produced and reviewed.] Arithmetic and date calculations in the references were checked with a script.

**Running.** A script posts each question to `POST /ask` with its own `asked_at` (and `history` for the multi-turn questions), one request at a time, and stores the answer, citations, tool calls, step timings and token usage as JSON.

**Judging correctness.** I compared each answer with its reference by hand, on the key facts (numbers, dates, conditions, the right source, no invented information):
- **Correct:** all key facts present and none wrong. Omitting an optional extra (for example the installment option) is still correct.
- **Partly correct:** no clearly wrong core fact, but a required fact is missing (for example the reason a promotion does not apply), or the answer leads with the wrong thing.
- **Wrong:** a wrong number, date, rule or conclusion, or an invented fact.

**Limits of this method.** One grader, a small set, and non-deterministic model output (see §5). Citations were checked by the code (every cited ID must have been returned by a tool in that request); I did not independently score citation precision.

## 3. Results by question

| ID | Topic | Verdict | Note |
|---|---|---|---|
| S01 | AC12N price, Đà Nẵng | Correct | List price and KM-2610 → 9.981.000 đ; installment option not mentioned |
| S02 | WP7R filter gift | Partly | Says no promotion; does not say KM-LOC ended on 20/10 |
| S03 | Compressor warranty | Wrong | Says 30-day activation window; handbook 3.2 (45 days for purchases from 01/10) should override the sheet |
| S04 | Fridge repair, Cần Thơ, Sunday | Correct | Ninh Kiều with Sunday hours |
| S05 | Return of online washing machine | Wrong | Says return is free within 7 days; the handbook charges a 10% fee and sets online-only/seal conditions |
| S06 | Copper pipe fee | Correct | 560.000 đ |
| S07 | iPhone 17 | Partly | Declines, but says the "policy document" has no information instead of stating the four product lines Lumen Home sells |
| S08 | "Nearest centre?" | Partly | Asks for province/district, but not product type, and lists every province |
| T01 | AC09N price, Hà Nội | Correct | 7.990.000 đ, KM-2610 → 7.191.000 đ |
| T02 | AC24P, 2M cap | Correct | 22.100.000 đ |
| T03 | AC12N on 31/10 23:30 | Correct | Promotion still valid on its last day: 9.891.000 đ |
| T04 | AC12N at 00:30 on 01/11 | Correct | Promotion ended; installment still available. (The test harness sent a +07:00 timestamp, so UTC parsing was not exercised.) |
| T05 | RF620SBS on 02/11 | Correct | KM-RF applied: 24.290.000 đ; non-stacking note omitted |
| T06 | WD10H, 14/10, trade-in | Correct | 18.890.000 đ (new price starts 15/10), 17.390.000 đ with trade-in |
| T07 | WP10H promotions | Correct | No installment (excluded), KM-LOC ended |
| T08 | Discontinued AC09S | Partly | No price and the hidden 2025 price not used; but states the 30-day activation window |
| T09 | Price for Huế | Correct | Not in region table; refers to hotline |
| T10 | RF360 activation, bought 10/10 | Wrong | 30 days instead of 45 (and a wrong resulting date) |
| T11 | WM10F, bought 20/09, asked 05/10 | Correct | 30-day window, deadline 20/10, still possible |
| T12 | Commercial use (laundry shop) | Wrong | Gives household warranty; misses the 6-month rule (3.4) |
| T13 | Store purchase, changed mind | Correct | Not returnable |
| T14 | Online return fee and deadline | Partly | 7-day window and 1.619.000 đ fee correct; does not say that 04/11 is the last day |
| T15 | Defect within 7 days | Correct | Exchange, with conditions |
| T16 | Sunday service, Hà Nội, washing machine | Correct | Only Cầu Giấy is open |
| T17 | Nha Trang centre closed | Partly | Lists the closed centre first, then the alternative |
| T18 | Pipe + travel fee | Correct | 760.000 đ |
| T19 | Technician late > 2 h | Wrong | Says 100.000 đ; handbook says a 200.000 đ voucher |
| T20 | Demand for 5M compensation | Partly | No cash compensation (right), but does not explain escalation (9.2) |
| T21 | Stock availability | Partly | Gives product status only and does not say stock is unknown |
| M1 | Follow-up: Đà Nẵng, Hải Châu, washing machine | Correct | Uses history; returns DN01 with hours |
| M2 | Follow-up: "Còn ở Hà Nội thì sao?" | Correct | Re-quotes AC12N for Hà Nội: 9.891.000 đ |
| M3 | Follow-up: "mua ở cửa hàng thì sao?" | Correct | Store purchases are not returnable for a change of mind |

## 4. Latency

Measured on 32 sequential requests. The LLM provider was on a free tier limited to 15 requests per minute.

| Metric | p50 | p95 | Max |
|---|---|---|---|
| Server-side latency | 5.7 s | 15.3 s | 26.5 s |
| Client-side latency | 5.8 s | 20.4 s | – |

- **Server-side** is the time inside `/ask`. The 7 requests above 9 s each contain a single LLM call of 4–9 s (one request has three such calls); typical calls take 1–2.5 s. I attribute these to provider variability on the free tier and did not investigate further.
- **Client-side** p95 can additionally include queueing from the 15 rpm limit (about 3 LLM calls per question), which sits outside the handler. Two earlier runs of the same code had server-side p95 of 8.4 s and 8.6 s, so p95 is sensitive to provider spikes; I report this run, not the best one.
- **LLM calls per question:** 3.1 on average (tool selection, optional follow-up rounds, final answer, validation retry). Typed tools and handbook search take 10–150 ms.
- **The validation retry fires on 32 of 32 requests** (mean 2.1 s). Without it, I estimate p50 ≈ 4.1 s and p95 ≈ 8.9 s. This is derived by subtracting recorded step times, not measured.
- **Targets:** p50 ≤ 3 s and p95 ≤ 7 s are **not met**.

## 5. Cost and stability

- **Tokens per question (average):** 8,185 input and 563 output.
- **Estimated cost per question:** (8,185 × [input price per token]) + (563 × [output price per token]) = [fill in from the provider's list price]. Actual spend during testing was $0 on the free tier.
- **Model choice.** `gemini-3.5-flash-lite` was chosen for latency and cost. The results show its limits: it sometimes ignores facts present in its context (T19, T10) and answers handbook questions from general knowledge when the right clause is not retrieved (S05).
- **Run-to-run variation.** Between two consecutive runs of the same code, verdicts differed on 7 of the 29 single-turn questions (S05, S07, T01, T11, T14, T20, T21). A single run is therefore indicative only; the 19/8/5 split above would move by a few questions on a rerun.
- **Full-handbook-in-prompt experiment.** One run placed the entire handbook in the prompt instead of retrieving clauses. Handbook-dependent answers did not improve (the model still gave 30 days, an invented return window and 100.000 đ), so I kept retrieval as the default and recorded the experiment as a next step.

## 6. Failure analysis

1. **Handbook retrieval returns the wrong clauses.** For most price questions the top hits are unrelated clauses (5.1, 1.1, 8.2); for the activation-window questions, clause 3.2 (a table) was not retrieved at all. Affects S03, S05, T10, T12, T19, T20.
2. **Precedence rule not enforced.** Clause 1.3 says the handbook overrides the spreadsheets; the model quotes the spreadsheet's 30 days (S03, T08, T10).
3. **Model reliability.** A wrong number although the clause was retrieved (T19) and inconsistent behaviour between runs.
4. **Validation retry on every request.** Adds about 2 s and an extra provider call; I have not yet identified why the first answer fails validation.
5. **Style drift.** Answers sometimes use "bạn/tôi", Markdown bold or inline cite IDs despite the prompt.
6. **History is passed to the model but not used for retrieval.** The three multi-turn questions passed because they were answered from spreadsheet tools or the model issued its own policy search, but the automatic handbook search uses only the current question, so a policy follow-up such as "and if it's opened?" could still miss the right clause.
7. **Tool-call loops.** In an earlier run a province written as "TP.HCM" did not match "TP. Hồ Chí Minh", and the agent retried the tool several times (7 LLM rounds, 54k tokens). Province normalisation is not implemented.

## 7. What I would do next

1. Diagnose the retry (log the raw first answer and exact validation error), and use short citation aliases if citation IDs are the cause.
2. Hybrid retrieval (BM25 + embeddings) with a query that includes recent turns; always include clauses that mention the source spreadsheets so that precedence reaches the model.
3. Test the handbook in a single system message or the user turn; evaluate a stronger model for the final answer and report cost against accuracy.
4. Province normalisation, a guard against repeated identical tool calls, and a lower step limit.
5. Repeat each question several times to measure variance, add more multi-turn tests, and evaluate on a paid-tier key to remove the rate-limit artefact from latency.
