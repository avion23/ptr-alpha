This adversarial review covers the **PTR Alpha** repository. Despite a high pass rate in the existing test suite (805 tests), the system contains several high-severity logic errors, data integrity risks, and mathematical contradictions.

---

### **Part 1: Confirmed Defects**

#### **1. Mathematical Logic Error: Log-Return / Simple-Return Mixing**
*   **Severity:** High (Distorts all rankings and signal scores)
*   **Exact Reference:** `src/analyzer/signals/core.py` (Line 38-42) and `src/analyzer/signals/assembly.py` (Line 46).
*   **Failure:** `core.py` calculates a weighted average of **logarithmic returns** (`np.log(w_vals[1:] / prev_vals)`). However, `assembly.py` treats the result as a **simple percentage return** by multiplying by 100 (`r_decayed_ret * 100`). Log returns are additive; simple returns are not. An "average log return" of 0.05 is not a 5% average return.
*   **Test/Fix:**
    *   *Test:* Create a price series with 100% volatility (e.g., $100 \to $200 \to $100). The average log return is 0, but simple returns are +100% and -50%.
    *   *Fix:* Convert the weighted log-return back to a simple return using `(np.exp(r_decayed_ret) - 1) * 100` in `assembly.py`.

#### **2. Data Integrity: Destructive "Replace" Logic in Reparsing**
*   **Severity:** Critical (Irreversible data loss)
*   **Exact Reference:** `src/analyzer/database.py`, `replace_transactions_for_docs` and `src/analyzer/parsing/metadata.py`, `consolidate_transactions`.
*   **Failure:** The pipeline follows a "Delete-then-Insert" pattern for documents. `consolidate_transactions` drops rows with invalid dates. If a re-parse uses an engine that misinterprets a date (OCR swap), that specific row is dropped. The `replace_transactions_for_docs` method deletes **all** existing rows for that `doc_id` and only inserts the surviving (potentially empty) subset. Valid historical data is deleted because the new parse was "cleaner" but incomplete.
*   **Test/Fix:**
    *   *Test:* Insert 10 rows for Doc A. Re-parse Doc A with a mock that returns only 5 rows. Observe that the DB now only has 5 rows.
    *   *Fix:* Implement a "high-water mark" check. If `new_row_count < existing_row_count * 0.8`, abort the transaction and move the document to an error queue for manual review.

#### **3. Security/Reliability: Brittle Ticker Extraction Blacklist**
*   **Severity:** Medium (False-positive signals)
*   **Exact Reference:** `src/analyzer/parsing/cells.py`, `_TICKER_BLACKLIST`.
*   **Failure:** The blacklist is a static `frozenset`. It is missing common OCR/Parsing artifacts that resemble tickers. Specifically, it includes "MARY" and "WISC" but omits "USD", "NAV", "DATE", and "TIME". Furthermore, `_extract_ticker` uses `re.search` for `$TICKER`. In House PDFs, a value like `$100K` will result in ticker `K` (Kellanova), creating highly weighted false signals for members "investing" in the letter K.
*   **Test/Fix:**
    *   *Test:* Pass "Account Value $100K" into `_extract_ticker`. Verify it returns `K`.
    *   *Fix:* Enhance `_TICKER_BLACKLIST` with currency codes and common units. Update the regex in `cells.py` to ensure `$TICKER` is followed by a word boundary or whitespace, and reject single-letter tickers unless they are in an explicit whitelist (e.g., `F`, `T`, `V`).

#### **4. Concurrency: Multi-Process DuckDB Write Lock**
*   **Severity:** Medium (Intermittent pipeline failure)
*   **Exact Reference:** `scripts/ocr_parallel.py`, `process_one`.
*   **Failure:** This script uses `ThreadPoolExecutor` (which is safe due to the single `db_writer` thread), but the `process_one` workers call `duckdb.connect(DB_PATH, read_only=True)`. In a multi-processing context (used by the main `parse` command via `Pool`), DuckDB will throw `duckdb.IOException: Process X holds a lock` if a write transaction is open in the main process.
*   **Test/Fix:**
    *   *Fix:* Use a singleton connection pool or ensure all metadata lookups are done in the main process before spawning workers.

---

### **Part 2: Contradictions and Logical Inconsistencies**

#### **1. The "Success" Parse Run Paradox**
*   **Contradiction:** `docs/house-data-parsing.md` states: "Parse-run records are written before the batch transaction replacement."
*   **Failure:** If `upsert_parse_run` succeeds (marking a document as `success`) but the subsequent `replace_transactions_for_docs` fails (due to a DB constraint or disk full), the database will report the document as successfully parsed while still containing either **no rows** or **stale rows**. This invalidates the parsing audit trail.
*   **Severity:** Medium (Audit trail corruption).

#### **2. Horizon/Frequency Mismatch in Validation**
*   **Contradiction:** `src/analyzer/validation.py` uses `periods_per_year = 365 / params.frequency_days` to annualize Sharpe.
*   **Failure:** This assumes a fixed-interval rebalance. However, the system is an "event-driven" backtest based on disclosure dates. If `frequency_days` is 30, but no trades occur for 3 months, the denominator in the Sharpe calculation is skewed.
*   **Severity:** Low/Medium (Statistical bias).

---

### **Part 3: Adversarial Hypotheses (High Risk)**

#### **1. The "Legislation Leak" Lookahead Bias (Hypothesis)**
*   **Ref:** `src/analyzer/signals/filters.py`, `_collapse_to_episodes`.
*   **Hypothesis:** The 14-day gap for collapsing episodes is meant to handle multi-lot trades. However, if a member trades on Day 1 (before news) and Day 10 (after news), and the disclosure for both happens on Day 20, the system collapses the "entry price" to the Day 1 price but assigns the "signal" to the Day 20 disclosure. This is technically correct (no lookahead on disclosure), but the **magnitude** of the signal is distorted because the return is calculated from a pre-disclosure price that might have already moved significantly.
*   **Risk:** Over-estimating member skill by giving them credit for price action that occurred *before* the public could react.

#### **2. Fragile Tesseract Orientation Detection**
*   **Ref:** `src/analyzer/parsing/ocr_parser.py`, `_orient_image`.
*   **Hypothesis:** The code uses `Rotate: (90|180|270)`. If Tesseract OSD returns `0` (or fails), and the form is a sideways checkbox form (common in 2021-2022), the parser attempts to read vertical text as horizontal.
*   **Risk:** `HOUSE_PARSER_AUDIT.md` shows 440 zero-row PDFs. Many of these are likely these legacy forms where orientation detection fails, meaning the system is blind to a specific class of high-volume traders.

---

### **Part 4: Missing Tests**

1.  **Ticker Collision Test:** No test exists for tickers that are also English words (e.g., "GAP", "TARGET", "BLOCK"). `cells.py` lacks alphanumeric boundary checks for company name mapping.
2.  **Delisting Scenario:** No test in `evaluate_backtest` checks what happens if a ticker is delisted *during* the horizon window but after the `as_of` date. Currently, it seems to fallback to the last known price, but it should ideally benchmark against a -100% loss or a cash-exit if the exit price is `NaN`.
3.  **BOM Encoding Test:** While `metadata.py` handles UTF-8 BOM, there is no test for **UTF-16** which the House Clerk occasionally uses for ZIP members.
