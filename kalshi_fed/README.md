# Kalshi Fed funds rate contracts

A daily time series of the market-implied fed funds target after each FOMC meeting, built from Kalshi's prediction-market contracts, July 2021 to September 2026.

Data as of **2026-09-27**: 52 FOMC meetings, 660 contracts, about 98,000 daily price bars, and 59 million contracts traded.

## Files

| File | What it is |
|---|---|
| `kalshi_fed_funds.ipynb` | The full pipeline: download, compute, charts, scorecard. Run top to bottom. |
| `fetch_kalshi_fed.py`, `build_series.py` | The same pipeline as scripts, plus they rebuild `fed_curve.html`. |
| `fed_curve.html` | Interactive version of the charts (self-contained page). |
| `data/implied_rate_daily.csv` | **Main output.** One row per meeting per day: implied upper bound, implied EFFR, outcome distribution, reliability flag. |
| `data/constant_horizon.csv` | One row per day: implied rate (upper bound and EFFR basis) for the next, 3rd and 6th upcoming meeting, the target in force, and actual EFFR. |
| `data/fred_effr.csv` | Daily EFFR and target upper bound from FRED. |
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

Looking backward from each meeting, the implied rate typically settles within 5bp of the eventual decision about **27 days** before the meeting (interquartile range 17–48 days). In other words, the outcome is usually priced in by the time of the previous month's data releases.

### Longer horizons are less accurate, and biased in a consistent direction

Error of the implied rate against the eventual decision, averaged over all reliable days:

| Horizon | Days | Mean absolute error | Mean error (implied − realized) |
|---|---|---|---|
| Next meeting (0–7 weeks) | 1,806 | 4.3bp | −1.0bp |
| 3rd meeting (~4 months) | 1,396 | 22.5bp | −10.6bp |
| 6th meeting (~9 months) | 432 | 34.5bp | −15.9bp |

A negative error means the market expected a lower rate than what happened. The bias follows the cycle:

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

### Where things stand (2026-09-27)

- Target upper bound: **4.00%**, set on Sep 16, 2026 after a 25bp hike.
- Actual EFFR: 3.88% (Sep 25), about 12bp below the upper bound.
- **Oct 28, 2026:** implied 4.16% upper bound, 4.04% on an EFFR basis, with about a 64% chance of another hike, 35% hold and 1% cut.
- **Dec 9, 2026:** implied 4.35% upper bound, 4.23% EFFR.
- Jan 2027 onward: ladders are too thinly quoted to price reliably today. The Jun 2027 through Jan 2028 meetings were only listed on Sep 18, 2026.
