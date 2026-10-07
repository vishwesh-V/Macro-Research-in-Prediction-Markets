-- Views and EDA queries for the macro release series (01b_eda_macro.py). Load kalshi_views.sql
-- first (for `all_markets`), then run this whole file's view block (everything before the first
-- `-- name:` line); the named queries follow. Paths are relative to the repo root.
--
-- Macro series are threshold ladders: one event per data release, contracts "above k" for several
-- strikes k. Every contract closes at 8:29 ET on the release day; an event's release date is its
-- latest close.


-- One row per macro contract, with its release date.
CREATE OR REPLACE VIEW macro_markets AS
SELECT
    *,
    CAST(max(close_time) OVER (PARTITION BY event_ticker) AT TIME ZONE 'America/New_York' AS DATE) AS release_date
FROM all_markets
WHERE series_ticker IN ('KXCPI', 'KXCPICORE', 'KXCPIYOY', 'KXPAYROLLS');


-- 1-minute candles of macro contracts. Sparse: a row exists only when the quote or a trade changed.
CREATE OR REPLACE VIEW macro_candles AS
SELECT
    c.ticker,
    m.event_ticker,
    m.series_ticker,
    m.release_date,
    m.strike,
    m.outcome,
    c.ts,
    c.bid_close AS bid,
    c.ask_close AS ask,
    (c.bid_close + c.ask_close) / 2 AS mid,
    c.ask_close - c.bid_close AS spread,
    c.volume
FROM 'data/kalshi_candles_1m.parquet' c
JOIN macro_markets m USING (ticker);


-- One row per macro contract per ET calendar day, from its first candle to its release (or today):
-- the quote in force at the end of each day, carried forward over days with no candle, plus that
-- day's volume. Every quote is kept, whatever its spread.
CREATE OR REPLACE VIEW macro_daily AS
WITH days AS (
    SELECT
        ticker,
        unnest(generate_series(
            min(CAST(ts AT TIME ZONE 'America/New_York' AS DATE)),
            least(any_value(release_date), current_date),
            INTERVAL 1 DAY
        ))::DATE AS date
    FROM macro_candles
    GROUP BY ticker
),
eod AS (  -- end of each ET day, as a UTC timestamp comparable with candles.ts
    SELECT ticker, date, (date + INTERVAL 1 DAY)::TIMESTAMP AT TIME ZONE 'America/New_York' AS day_end
    FROM days
),
daily_volume AS (
    SELECT ticker, CAST(ts AT TIME ZONE 'America/New_York' AS DATE) AS date, sum(volume) AS volume
    FROM macro_candles
    GROUP BY ALL
)
SELECT
    c.ticker,
    c.event_ticker,
    c.series_ticker,
    c.release_date,
    c.strike,
    c.outcome,
    e.date,
    date_diff('day', e.date, c.release_date) AS days_to_release,
    c.bid,
    c.ask,
    c.mid,
    c.spread,
    coalesce(v.volume, 0) AS volume,
    c.ts AS quote_ts
FROM eod e
ASOF JOIN macro_candles c ON e.ticker = c.ticker AND e.day_end > c.ts
LEFT JOIN daily_volume v ON v.ticker = e.ticker AND v.date = e.date;


-- name: coverage
-- One row per series: what Kalshi lists and how much of it is downloaded.
WITH t AS (SELECT ticker, count(*) AS trades FROM 'data/kalshi_trades.parquet' GROUP BY ticker),
     k AS (SELECT ticker, count(*) AS candles FROM 'data/kalshi_candles_1m.parquet' GROUP BY ticker)
SELECT
    m.series_ticker,
    count(DISTINCT m.event_ticker) AS releases,
    count(*) AS contracts,
    count(*) FILTER (m.status = 'finalized') AS settled,
    min(m.release_date) AS first_release,
    max(m.release_date) AS last_release,
    sum(m.volume) AS volume,
    count(*) FILTER (k.candles > 0) AS contracts_with_data,
    coalesce(sum(t.trades), 0) AS trades_downloaded,
    coalesce(sum(k.candles), 0) AS candles_downloaded
FROM macro_markets m
LEFT JOIN t USING (ticker)
LEFT JOIN k USING (ticker)
GROUP BY ALL
ORDER BY m.series_ticker;


-- name: releases
-- One row per release: the ladder's strikes and where the print landed. A settled print lies above
-- the highest strike that settled YES and at or below the lowest strike that settled NO.
SELECT
    series_ticker,
    event_ticker,
    release_date,
    count(*) AS strikes,
    min(strike) AS lowest_strike,
    max(strike) AS highest_strike,
    bool_and(status = 'finalized') AS settled,
    max(strike) FILTER (result = 'yes') AS print_above,
    min(strike) FILTER (result = 'no') AS print_at_most,
    sum(volume) AS volume
FROM macro_markets
GROUP BY ALL
ORDER BY series_ticker, release_date DESC;


-- name: liquidity_by_month
-- Monthly traded contracts per series.
SELECT
    m.series_ticker,
    date_trunc('month', t.created_time) AS month,
    sum(t.count) AS contracts,
    count(*) AS trades,
    median(t.count) AS median_trade_size
FROM 'data/kalshi_trades.parquet' t
JOIN macro_markets m USING (ticker)
GROUP BY ALL
ORDER BY m.series_ticker, month;


-- name: probability_path
-- End-of-day mid for every strike of one release. Parameter $event_ticker.
SELECT date, days_to_release, ticker, strike, outcome, bid, ask, mid, spread, volume
FROM macro_daily
WHERE event_ticker = $event_ticker
ORDER BY date, strike;


-- name: spread_by_price
-- Median end-of-day spread per series and price level (10¢ bins of the mid), 1–60 days before the
-- release.
SELECT
    series_ticker,
    least(floor(mid * 10) / 10, 0.9) AS price_bin,
    count(*) AS contract_days,
    median(spread) AS median_spread
FROM macro_daily
WHERE days_to_release BETWEEN 1 AND 60 AND mid IS NOT NULL
GROUP BY ALL
ORDER BY series_ticker, price_bin;


-- name: ladder_violations
-- A ladder is consistent when P(above k) falls as k rises. Per series: the share of release-days
-- (1–60 days out, at least two strikes quoted) where some higher strike is priced above a lower one,
-- by more than 0¢ and by more than 2¢.
WITH d AS (
    SELECT
        series_ticker,
        event_ticker,
        date,
        mid,
        lead(mid) OVER (PARTITION BY event_ticker, date ORDER BY strike) AS next_mid
    FROM macro_daily
    WHERE days_to_release BETWEEN 1 AND 60 AND mid IS NOT NULL
),
day AS (
    SELECT
        series_ticker,
        event_ticker,
        date,
        count(next_mid) AS pairs,
        bool_or(next_mid > mid) AS any_violation,
        bool_or(next_mid > mid + 0.02) AS violation_over_2c
    FROM d
    GROUP BY ALL
)
SELECT
    series_ticker,
    count(*) AS release_days,
    avg(any_violation::INT) AS share_any_violation,
    avg(violation_over_2c::INT) AS share_violation_over_2c
FROM day
WHERE pairs > 0
GROUP BY ALL
ORDER BY series_ticker;
