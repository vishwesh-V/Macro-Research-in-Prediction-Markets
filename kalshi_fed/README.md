# Kalshi Fed funds rate contracts

A daily time series of the market-implied fed funds target after each FOMC meeting, built from Kalshi's prediction-market contracts, July 2021 to September 2026.

Data as of **2026-09-27**: 52 FOMC meetings, 660 contracts, about 98,000 daily price bars, and 59 million contracts traded.

## Files

| File | What it is |
|---|---|
| `kalshi_fed_funds.ipynb` | The full pipeline: download, compute, charts, scorecard. Run top to bottom. |
| `fetch_kalshi_fed.py`, `build_series.py` | The same pipeline as scripts, plus they rebuild `fed_curve.html`. |
| `fed_curve.html` | Interactive version of the charts (self-contained page), including a method-and-validation section with the replication, the ablation and the survey benchmark. |
| `data/implied_rate_daily.csv` | **Main output.** One row per meeting per day: implied upper bound, implied EFFR, outcome distribution, reliability flag. |
| `data/constant_horizon.csv` | One row per day: implied rate (upper bound and EFFR basis) for the next, 3rd and 6th upcoming meeting, the target in force, and actual EFFR. |
| `data/fred_effr.csv` | Daily EFFR and target upper bound from FRED. |
| `data/trades.csv.gz` | Every trade on every contract, 2021–2026 (about 222,000 trades). |
| `data/implied_rate_daily_dkw.csv` | Daily implied rates rebuilt with the Diercks–Katz–Wright method, from the trade data. |
| `data/method_comparison.csv` | Our method vs theirs, scored against realized decisions. |
| `data/nyfed_sme.csv` | NY Fed dealer survey: median modal forecast for each upcoming meeting, 36 surveys. |
| `data/survey_comparison.csv` | Survey vs Kalshi forecasts on each survey's due date. |
| `data/ablation.csv` | The 2×2 ablation: each pricing rule paired with each monotonicity rule. |
| `data/validation.json` | Replication and benchmark results, read by `build_series.py` for the website. |
| `data/track_record.csv` | Probability the market put on the actual decision 1 day, 1 week and 1 month before each settled meeting. |
| `data/candles_daily.csv` | Raw daily bid, ask, last trade, volume and open interest for every contract. |
| `data/markets.csv` | One row per contract with strike and settlement result. |
| `data/meeting_summary.csv` | Scorecard: implied rate the day before each meeting vs the decision. |
| `data/realized_target.csv` | Upper bound after each settled meeting. |

To run: open the notebook and run all cells. It needs only `pandas` and `matplotlib`. The first run downloads from Kalshi's public API (no key needed, about 3 minutes) and caches the results in `data/`. Set `REFRESH = True` to download again. Script version: `python3 fetch_kalshi_fed.py && python3 build_series.py` (`--rebuild` recomputes from the cached CSVs).

## The contracts

Kalshi series `KXFED` ("Fed funds rate") has one event per FOMC meeting, for example `KXFED-26OCT`. Events before 2026 are listed under the older `FED-*` tickers in the same series. Each event is a ladder of yes/no contracts:

> Will the upper bound of the federal funds rate be above *k*% following the Fed's Oct 28, 2026 meeting?

Strikes are 25bp apart. A contract pays $1 if the published upper bound is above *k*, so its price is the market's probability P(upper bound > *k*). Ladders have grown over time: early 2021 meetings had a single strike, 2022–2025 meetings had 7 to 19, and 2027 meetings have 25.

Data comes from the Kalshi Trade API v2 ([spec](https://docs.kalshi.com/openapi.yaml)):

- `GET /events?series_ticker=KXFED`: list of meetings.
- `GET /markets?event_ticker=…` and `GET /historical/markets?event_ticker=…`: contracts. Markets that settled before Kalshi's historical cutoff (2026-07-29 at the time of download) are only served by the historical endpoints.
- `GET /series/KXFED/markets/{ticker}/candlesticks` and `GET /historical/markets/{ticker}/candlesticks`: daily bars (`period_interval=1440`). The two endpoints name fields differently (`close_dollars` vs `close`), and the code handles both.

## Approach

**1. One price per contract per day.**
- Use the closing bid/ask midpoint when the spread is 20¢ or less.
- If the spread is 20–40¢, use the last trade, clipped into the bid/ask. A stale trade can't sit outside the live quote.
- If the spread is wider than 40¢, drop that strike for the day. A quote like 1¢/99¢ carries no information.

**2. Fill quiet days.** Kalshi only emits a daily bar when something changes, so each contract's last quote is carried forward to every calendar day until its meeting (or the data date). Without this, strikes drop in and out from day to day and the implied rate jumps for no reason.

**3. Make the ladder consistent.** P(rate > *k*) has to fall as *k* rises. Raw prices sometimes violate this, so a weighted isotonic regression (pool-adjacent-violators) fits the closest non-increasing curve. Each price is weighted by 1/(spread + 2¢), so tight quotes dominate. An earlier running-minimum version let one stale low price drag every higher strike down with it.

**4. Turn the ladder into a distribution and a mean.** Differencing adjacent strikes gives the probability of each outcome:
- P(upper bound ≤ *k*₀) = 1 − P(> *k*₀), placed at *k*₀.
- P(*k*ᵢ < upper bound ≤ *k*ᵢ₊₁) = P(> *k*ᵢ) − P(> *k*ᵢ₊₁), valued at the middle of the 25bp outcomes in that range (just *k*ᵢ + 0.25 when strikes are adjacent).
- P(upper bound > *k*ₙ), placed at *k*ₙ + 0.25.

The **implied upper bound** is the probability-weighted average. The **most likely** outcome is the bucket with the most probability.

**5. Flag unreliable days.** A meeting-day is marked `reliable = False` when any of these is more than 10%:
- probability below the lowest usable strike. Below 0.25% is exempt, because the zero lower bound makes it exact.
- probability above the highest usable strike.
- probability in gaps wider than 50bp between usable strikes.

In those cases the mean depends on where the unplaced probability is assumed to sit. Unreliable rows stay in `implied_rate_daily.csv`, but are left out of `constant_horizon.csv` and the history chart. Overall, 63% of meeting-days are reliable.

**6. Realized decisions.** The upper bound after each meeting is read from how its ladder settled: the highest strike that resolved "yes", plus 25bp.

**7. Constant-horizon series.** For each day, the next, 3rd and 6th upcoming meetings are identified, and their implied rates are recorded that day. Because FOMC meetings are about 6–7 weeks apart, "3rd meeting" is roughly 4 months out and "6th meeting" roughly 9 months. The horizon is counted in meetings, not fixed calendar days.

**8. Implied effective rate (EFFR).** Kalshi settles on the target's upper bound. Fed funds futures, and most rates research, use the effective rate: the volume-weighted rate banks actually pay, which trades inside the range. To put Kalshi on that basis:

`implied_effr = implied_upper_bound + (EFFR − upper bound)`

The gap is the median of the last 20 published days as of each date, so each row only uses information available then. The data comes from FRED series `EFFR` and `DFEDTARU`. The gap averaged −17bp from 2021 through 2024, −16bp in 2025 and −12bp in 2026, as bank reserves got scarcer.

**9. Hair chart.** On the first trading day of each month, the implied upper bound for every reliably priced upcoming meeting is joined into one path, starting from the target in force that day. Drawing all of these against the realized target shows every forecast at once.

## Assumptions and limitations

- **The target is on the 25bp grid.** Outcomes between strikes are assumed to be the 25bp steps in between. An off-grid move (for example 10bp) would be misplaced.
- **Prices are probabilities.** Contract prices are treated as risk-neutral probabilities, with no adjustment for fees, a risk premium, or the favorite–longshot bias common in prediction markets. Near-certain outcomes trade at 98–99¢, not $1, so there's a small floor of probability on unlikely outcomes.
- **Tail placement.** Probability beyond the ladder is placed at its edge (the lowest strike, or the top strike + 25bp). This understates the mean when hikes outrun the ladder, and overstates it when cuts outrun it. The reliability flag catches the cases where this matters by more than 10% probability.
- **Daily closes.** Bars close at midnight ET, so each day is an end-of-day snapshot. Intraday moves (a CPI release, the minutes after an FOMC statement) are only seen through that day's close. Pass `INTERVAL = 60` or `1` for hourly or minute bars.
- **Stale quotes.** Carrying quotes forward assumes an untouched quote is still live. For thinly traded ladders that can hold a price for weeks.
- **Liquidity varies a lot.** Meetings more than about 4 months out are often quoted with spreads of 50–98¢. That is why the 6th-meeting series covers only 25% of days, and why meetings from March 2027 on don't have reliable values yet.
- **EFFR gap held constant.** The implied EFFR assumes today's EFFR − upper bound gap holds through every future meeting. It has moved by about 5bp over two years, so the implied EFFR for distant meetings carries a few basis points of extra uncertainty.
- **Meeting date.** The meeting date is taken from each contract's close time. For some 2021 contracts this is the day before the decision.

## Results

### The contracts track the Fed closely near each meeting

On the last day before each of the 41 settled meetings:

| Metric | Value |
|---|---|
| Most-likely outcome matched the decision | **41 of 41** |
| Implied rate within 5bp of the decision | 39 of 41 |
| Median absolute error | 0.6bp |
| Mean absolute error | 1.3bp |
| Largest error | +10.5bp (Sep 2024) |

The biggest miss was **September 2024**, the first cut of the cycle. The day before, the market put 54% on a 50bp cut and 43% on 25bp, for an implied 5.10%. The Fed cut 50bp, to 5.00%. The meeting explorer in the notebook shows the market swinging between the two outcomes through August and September. The next-largest miss was July 2026 (+6.6bp): the market gave a 25bp hike a 25% chance, and the Fed held at 3.75%.

Probability the market put on the decision the Fed actually made:

| When | Median | Meetings at ≥90% | Meetings below 50% |
|---|---|---|---|
| 1 day before | 97% | 85% | 0 of 41 |
| 1 week before | 95% | 68% | 2 of 41 |
| 1 month before | 81% | 27% | 5 of 39 |

The two meetings below 50% a week out are the two genuine surprises of the period. **June 2022 (75bp hike):** 5% a week before, 80% the day before, after press reports in the final days. **September 2024 (50bp cut):** 5% a week before, 54% the day before. The odds-by-meeting chart shows any meeting's odds 1 month, 1 week and 1 day out, with the actual decision marked.

Looking backward from each meeting, the implied rate typically settles within 5bp of the eventual decision about **27 days** before the meeting (interquartile range 17–48 days). In other words, the outcome is usually priced in by the time of the previous month's data releases.

### Longer horizons are less accurate, and biased in a consistent direction

Error of the implied rate against the eventual decision, averaged over all reliable days:

| Horizon | Days | Mean absolute error | Mean error (implied − realized) |
|---|---|---|---|
| Next meeting (0–7 weeks) | 1,806 | 4.3bp | −1.0bp |
| 3rd meeting (~4 months) | 1,396 | 22.5bp | −10.6bp |
| 6th meeting (~9 months) | 432 | 34.5bp | −15.9bp |

A negative error means the market expected a lower rate than what happened. The forecast-error histograms in the notebook and website show the full distributions. Next-meeting errors are tightly bunched around zero. The 3rd- and 6th-meeting errors spread over roughly ±60bp and are shifted left, with a long left tail from 2022. The bias follows the cycle:

- **2022 (hiking cycle):** the market consistently underpriced how far the Fed would go. The 3rd-meeting forecast averaged 38bp too low, and the 6th-meeting forecast 55bp too low. The ladder also ran out of strikes during this period, so many days are flagged unreliable.
- **2023 (hold at 5.25–5.50%):** 3rd-meeting forecasts were nearly unbiased (−3bp). 6th-meeting forecasts were 38bp too low, because the market kept pricing cuts that didn't come.
- **2024–2025 (cutting cycle):** close to unbiased at every horizon (−8bp to +2bp).
- **2026:** the market again underpriced tightening. In 2026 the 3rd-meeting forecast averaged 17bp too low, ahead of the September 2026 hike to 4.00%.

This matches the well-known pattern in fed funds futures: forecasts beyond the next couple of meetings underreact to turning points. We haven't compared Kalshi against futures directly yet; the implied EFFR column puts the two on the same basis for that.

### Every implied path at once

The hair chart makes the same point visually:

- **2022:** almost every hair falls below the target line. For the December 2022 meeting, the path priced in March 2022 expected 1.65%, June 2.63% and September 3.84%. The Fed went to 4.50%.
- **2023:** hairs repeatedly bend down from the 5.50% plateau, pricing cuts that took until September 2024 to arrive.
- **2024–2025:** hairs mostly lie on top of the cutting path.
- **2026:** hairs point down toward about 3% in 2027 while the Fed hiked to 4.00%. Only the most recent paths turn up.

## Replication of Diercks, Katz & Wright (2026)

Diercks, Katz and Wright, *Kalshi and the Rise of Macro Markets* (NBER Working Paper 34702; Federal Reserve FEDS 2026-010), build daily fed funds distributions from the same Kalshi ladders. Their method differs from ours in two places: they price each contract at its **last trade** (carried forward), where we use the bid/ask midpoint with fallbacks; and they enforce monotonicity **outward from the mode**, where we use a weighted isotonic fit. Notebook section 13 rebuilds their method from the trade-level data (last trade of each ET day) and scores both methods on the same meeting-days: 5,934 days between 1 and 160 days before each of the 41 settled meetings.

| | Their method | Ours |
|---|---|---|
| Mode right, 1 day before | **41 of 41** | **41 of 41** |
| Mean absolute error of the mean, 1 day before | 1.6bp | 1.2bp |
| … 1 week before | 3.0bp | 2.7bp |
| … 1 month before | 5.9bp | 5.9bp |
| … 3 months before | 21.9bp | 19.3bp |
| … averaged over 1–160 days | 20.8bp | 17.9bp |
| Mode, averaged over 1–160 days | 21.0bp | 16.1bp |
| Average day-to-day change in the implied rate | 1.6bp | 1.3bp |

- **The replication succeeds.** Their headline result, a perfect day-before record for the mode, holds under their method on our data, including on their 2022–2025 window, and also for 2021 and 2026, which their paper doesn't cover.
- **Inside a month they are about equal.** From roughly 45 to 20 days out their mode is slightly more accurate than ours (about 1bp).
- **Beyond a month ours is more accurate and less noisy**, by 3–5bp on average and by up to 25bp at 120–160 days. That's where the difference between the methods matters: the median last trade behind their prices is 3–6 days old, and on thinly traded strikes much older.
- **Which choice matters (2×2 ablation).** Pairing each pricing rule with each monotonicity rule shows the gain comes from pricing. With their mode-outward rule, switching from last trades to quotes cuts the 1–160 day error from 20.8 to 18.0bp; with quotes, the monotonicity rule barely matters (18.0 vs 17.9bp). With last trades, their mode-outward rule is the better one near the meeting (1.6 vs 2.8bp the day before). About 40–45% of last-trade prices are more than a week old at every horizon. Our quotes + isotonic cell reproduces the main series exactly.
- **Selection check.** Restricting both methods to the days our reliability flag passes lowers both errors but keeps the gap (17.0bp vs 14.0bp), so our advantage doesn't come from the filter.

The time of day for the last trade isn't specified in their paper; we assume end of ET day. The gaps have not been tested for statistical significance.

## Benchmark: NY Fed Survey of Market Expectations

Before each FOMC meeting the New York Fed surveys primary dealers on their most likely target rate after each upcoming meeting. Notebook section 14 downloads 36 dealer surveys from March 2021 to July 2026: the spreadsheets from July 2023, and the text PDFs before that. The November 2022 to June 2023 results are image-only PDFs and can't be read, and three 2021 PDFs use a different layout, so those surveys are missing. Each survey forecast is compared with Kalshi on the survey's due date. The sample is limited to meetings whose ladder is reliably priced that day, which leaves 132 forecasts from 33 surveys covering 37 meetings.

Mean absolute error against the realized upper bound (bp):

| Horizon | Forecasts | Dealer survey | Kalshi mode (ours) | Kalshi mean (ours) | Kalshi mean (DKW) |
|---|---|---|---|---|---|
| ≤ 45 days | 33 | 1.5 | 1.5 | 2.5 | 2.9 |
| 46–90 days | 30 | 8.3 | 6.7 | 10.6 | 11.0 |
| 91–180 days | 43 | 23.3 | 21.5 | 26.8 | 29.8 |
| > 180 days | 26 | 47.1 | 41.3 | 49.6 | 58.0 |
| All | 132 | 19.1 | **17.0** | 21.5 | 24.4 |

- **Kalshi's mode matches or beats the dealer survey at every horizon.** The survey is a modal forecast, so the mode is the like-for-like comparison. It was exactly right 59% of the time, against 57% for the survey.
- **Kalshi's mean does worse than the survey.** The mean spreads probability over outcomes that don't happen, which costs accuracy against a single realized value.
- This is consistent with Diercks–Katz–Wright's finding that Kalshi is comparable to or better than surveys, here on a longer sample. With 132 overlapping forecasts the differences are suggestive, not statistically established.

### Where things stand (2026-09-27)

- Target upper bound: **4.00%**, set on Sep 16, 2026 after a 25bp hike.
- Actual EFFR: 3.88% (Sep 25), about 12bp below the upper bound.
- The odds for Oct 28 moved a lot in a month. On Aug 28, before the September hike, the market put 41% on 3.75%, 51% on 4.00% and 5% on 4.25% or higher. Now it's 35% on 4.00% and 64% on 4.25% or higher.
- **Oct 28, 2026:** implied 4.16% upper bound, 4.04% on an EFFR basis, with about a 64% chance of another hike, 35% hold and 1% cut.
- **Dec 9, 2026:** implied 4.35% upper bound, 4.23% EFFR.
- Jan 2027 onward: ladders are too thinly quoted to price reliably today. The Jun 2027 through Jan 2028 meetings were only listed on Sep 18, 2026.
