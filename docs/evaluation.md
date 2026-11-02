# Evaluation report

## 1. Summary

This report records an earlier evaluation of the on-demand `search_policy` configuration with typed tools over PostgreSQL and `gemini-3.5-flash-lite`. The current implementation also leaves handbook search on demand, with no automatic pre-search; the model may request focused retrieval when it needs policy evidence. Results below describe the evaluated run and are not a guarantee of current accuracy or latency. The set contained the 8 supplied sample questions and 24 additional questions (32 total, including 3 multi-turn questions that pass a `history` list).

| | Count | Share |
|---|---|---|
| Correct | 24 | 75% |
| Partly correct | 6 | 19% |
| Wrong | 2 | 6% |

- **Questions answered from the spreadsheets (price, promotion, service centres, fees): 13 correct, 3 partly correct, 1 wrong** (17 questions). Date-sensitive rules (price-row effective dates, inclusive promotion end date, the 2,000,000 đ cap, a timestamp just after midnight, temporary closures, the hidden 2025 price sheet) were handled correctly.
- **Questions that depend on the handbook or on the limits of the materials: 9 correct, 2 partly correct, 1 wrong** (12 questions). The handbook-over-spreadsheet conflict on the 30/45-day activation window was resolved correctly in one of the three questions that touch it (S03); T08 and T10 still quote the spreadsheet's 30 days.
- **Multi-turn questions: 2 correct, 1 partly correct** out of 3.
- **Latency is close to, but does not meet, the target.** Server-side p50 is 4.2 s and p95 is 18.8 s (target ≤ 3 s / ≤ 7 s). 27 of 32 requests finished within 7 s; the p95 is set by two requests that each contained single LLM calls of 10–13 s (provider variability on a free-tier key).

## 2. Method

**Test set.** 32 questions: the 8 samples (S01–S08), 21 single-turn questions of my own (T01–T21) and 3 multi-turn questions (M1–M3). My own questions cover: price rows that change over time, promotion caps and inclusive end dates, a timestamp just after midnight, a discontinued model and the hidden 2025 price sheet, a province missing from the price table, handbook-versus-spreadsheet conflicts (30 vs 45 day activation window), commercial use, returns (online vs store, deadline arithmetic), service centres (Sunday hours, temporary closure), fee arithmetic, compensation and escalation, and questions the materials cannot answer (stock availability, out-of-scope products).

**Reference answers.** For each question I prepared a reference answer from the spreadsheets and handbook, noting the relevant sheet/row or clause and the facts that must appear. Arithmetic and date calculations in the references were checked with a script.

**Running.** A script posts each question to `POST /ask` with its own `asked_at` (and `history` for the multi-turn questions), one request at a time, and stores the answer, citations, tool calls, step timings and token usage as JSON.

**Judging correctness.** I compared each answer with its reference by hand, on the key facts (numbers, dates, conditions, the right source, no invented information):
- **Correct:** all key facts present and none wrong. Omitting an optional extra is still correct.
- **Partly correct:** no clearly wrong core fact, but a required fact is missing (for example the reason a promotion does not apply), or a minor detail is wrong.
- **Wrong:** a wrong number, date, rule or conclusion, or an invented fact.

**Limits of this method.** One grader, a small set, and non-deterministic model output (see §5). The final configuration was run once on the full set. Citations were checked by the code (every cited ID must have been returned by a tool in that request); I did not independently score citation precision.

## 3. Results by question

| ID | Topic | Verdict | Note |
|---|---|---|---|
| S01 | AC12N price, Đà Nẵng | Correct | List price, KM-2610 → 9.981.000 đ, installment conditions |
| S02 | WP7R filter gift | Partly | Says no promotion; does not say KM-LOC ended on 20/10 |
| S03 | Compressor warranty | Correct | 120 months; 30 days for purchases before 01/10, 45 days from 01/10; 60 months otherwise |
| S04 | Fridge repair, Cần Thơ, Sunday | Correct | Ninh Kiều with Sunday hours |
| S05 | Return of online washing machine | Correct | 7 days, sealed and not installed, 10% fee, refund in 5–7 working days |
| S06 | Copper pipe fee | Correct | 560.000 đ |
| S07 | iPhone 17 | Correct | Declines; lists the four product lines Lumen Home sells |
| S08 | "Nearest centre?" | Partly | Asks for province/district, but not the product type |
| T01 | AC09N price, Hà Nội | Correct | 7.990.000 đ, KM-2610 → 7.191.000 đ |
| T02 | AC24P, 2M cap | Correct | 22.100.000 đ |
| T03 | AC12N on 31/10 23:30 | Correct | Promotion still valid on its last day: 9.891.000 đ |
| T04 | AC12N at 00:30 on 01/11 | Correct | No promotion; installment still available. (The test harness sent a +07:00 timestamp, so UTC parsing was not exercised.) |
| T05 | RF620SBS on 02/11 | Correct | KM-RF applied: 24.290.000 đ; notes that discounts are not combined |
| T06 | WD10H, 14/10, trade-in | Correct | 18.890.000 đ (new price starts 15/10), 17.390.000 đ with trade-in |
| T07 | WP10H promotions | Correct | No installment (excluded), KM-LOC ended |
| T08 | Discontinued AC09S | Partly | No price and the hidden 2025 price not used; but states the 30-day activation window |
| T09 | Price for Huế | **Wrong** | "Thừa Thiên Huế" is not in the region table, and the model then supplied the region itself ("Miền Trung") and quoted 10.590.000 đ instead of referring the customer to the hotline |
| T10 | RF360 activation, bought 10/10 | **Wrong** | 30 days instead of 45; in this run the model did not request the optional handbook search |
| T11 | WM10F, bought 20/09, asked 05/10 | Correct | 30-day window, still possible |
| T12 | Commercial use (laundry shop) | Correct | 6 months, no inverter-motor term |
| T13 | Store purchase, changed mind | Correct | Not returnable |
| T14 | Online return fee and deadline | Correct | 04/11 is exactly day 7, fee 1.619.000 đ, refund timing |
| T15 | Defect within 7 days | Correct | Exchange, with conditions |
| T16 | Sunday service, Hà Nội, washing machine | Correct | Only Cầu Giấy is open |
| T17 | Nha Trang centre closed | Correct | Reports the closure and refers to Cam Ranh (no address hours for Cam Ranh) |
| T18 | Pipe + travel fee | Correct | 760.000 đ |
| T19 | Technician late > 2 h | Correct | 200.000 đ voucher, once per request |
| T20 | Demand for 5M compensation | Partly | No cash compensation (right), but does not explain escalation (9.2) |
| T21 | Stock availability | Partly | Gives product status only and does not say stock is unknown |
| M1 | Follow-up: Đà Nẵng, Hải Châu, washing machine | Partly | Uses history and returns DN01, but states the weekday hours as Monday–Friday instead of Monday–Saturday |
| M2 | Follow-up: "Còn ở Hà Nội thì sao?" | Correct | Re-quotes AC12N for Hà Nội: 9.891.000 đ |
| M3 | Follow-up: "mua ở cửa hàng thì sao?" | Correct | Store purchases are not returnable for a change of mind |

## 4. Latency

Measured on 32 sequential requests. The LLM provider was on a free tier.

| Metric | Value |
|---|---|
| Server-side p50 | 4.2 s |
| Server-side p95 | 18.8 s |
| Server-side max | 35.8 s |
| Requests finishing within 3 s / 5 s / 7 s | 5 / 22 / 27 of 32 |
| Excluding the two slowest requests: p50 / p95 | 4.0 s / 10.2 s |

- **Where the time goes.** There are 2.4 LLM calls per question on average (tool selection, optional follow-up rounds, final answer). The median LLM call takes 1.7 s; typed tools and handbook searches, when requested, take about 45 ms at the median (max 314 ms). Latency is therefore almost entirely LLM calls.
- **What sets the p95.** The two slowest requests (24.9 s and 35.8 s) each contain single LLM calls of 10–13 s, while the same kinds of call usually take 1–2 s. I attribute this to provider variability and did not investigate further.
- **Targets:** p50 ≤ 3 s and p95 ≤ 7 s are **not met** (4.2 s / 18.8 s).
- **Client-side latency** is within a few milliseconds of server-side latency in this run.

## 5. Cost, stability and design comparison

- **Tokens per question (average):** 7,251 input and 226 output.
- **Provider cost:** Not estimated; testing used the provider's free tier, so actual spend was $0.
- **Model choice.** `gemini-3.5-flash-lite` was chosen for latency and cost. The evaluated run's remaining errors include a missed optional handbook search (T10) and an improvised region (T09).
- **Handbook access: three configurations tried on the same questions.**

| Configuration | Result |
|---|---|
| A. Forced initial handbook search on every question, injected into the prompt | 19 correct / 8 partly / 5 wrong; an extra validation-retry LLM call on 32 of 32 requests; p50 5.7 s |
| B. Entire handbook in the prompt (one run, 29 questions) | Handbook-dependent answers did not improve; one request failed with HTTP 502 |
| C. On-demand `search_policy` tool (evaluated configuration) | 24 correct / 6 partly / 2 wrong; no validation retries; p50 4.2 s |

  In A the retrieved clauses were often unrelated to the question and the model sometimes answered from general knowledge. In C the model searches with its own focused query (for example "đổi trả máy giặt mua online chưa mở thùng") and answers from the clauses it gets. The current implementation retains this on-demand tool and does not run A's forced initial search. Other configuration differences between these runs may also affect the comparison, so the results should not be read as a controlled comparison of retrieval alone.
- **Run-to-run variation.** Between two consecutive runs of configuration A, verdicts differed on 7 of 29 questions. The final configuration was run once on the full set, so its result is indicative only.

## 6. Failure analysis

1. **Guessing a missing region (T09).** When a province is not in the price-region table, the model retried with a region of its own choosing. The tool should refuse and tell the model to refer the customer to the hotline.
2. **Optional handbook search can be missed (T10, T08).** For a warranty question about a purchase after 01/10 the model read the spreadsheet only and gave the sheet's 30 days; the handbook (clause 1.3: handbook wins; clause 3.2: 45 days) was not consulted. When it did search (S03), it answered correctly. Because the current design has no automatic fallback search, tool selection and its prompt guidance remain important.
3. **Missing explanations (S02, T20, T21, S08).** Answers are correct but omit a reason, an escalation step or a clarifying detail.
4. **Latency spikes.** Single LLM calls of 10–13 s on the free tier push p95 over target.
5. **History is passed to the model, not directly to retrieval.** The multi-turn questions behaved correctly (one small hours error), but `search_policy` receives the focused query the model constructs; I have not tested long conversations or whether follow-up context is consistently reflected in that query.
6. **Tool-call loops (earlier runs).** A province written as "TP.HCM" did not match "TP. Hồ Chí Minh", and the agent retried the tool several times. Province normalisation is not implemented.

## 7. What I would do next

1. Make the price tool return an explicit "province not in the region table; do not guess" result. Evaluate whether prompt guidance or query construction can make the model request handbook evidence for policy-dependent warranty, return, and fee questions without bringing back an unselective search on every request.
2. Normalise province names, guard against repeated identical tool calls, and keep the step limit low.
3. Run each question several times to measure variance, add more multi-turn and adversarial tests, and measure latency on a paid-tier key.
4. Evaluate a stronger model for the final answer and report accuracy against cost and latency.
5. Hybrid retrieval (BM25 + embeddings) for handbook search, if the handbook grows beyond a few pages.
