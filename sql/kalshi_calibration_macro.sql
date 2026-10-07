-- Queries for 03_calibration_macro.py. Load kalshi_views.sql first (for `all_markets`) and
-- define `candles_raw` and `trades_raw` views over the 1-minute candle and trade files.
--
-- Macro series (CPI, payrolls) are threshold ladders: each release is one event with
-- contracts "above k" for several strikes k, several of which can settle YES. Every
-- contract closes at 8:29 ET on the release day, a minute before the 8:30 release; the
-- release date of an event is its latest close.


-- name: forecasts
-- One row per settled contract and horizon: the quote in force at $snapshot_hour ET, `ttm`
-- days before the release, and whether the contract settled YES (`y`). Contracts not yet
-- listed, or already closed, at a snapshot drop out.
--   p               bid/ask midpoint (ladder contracts are not exclusive, so no rescaling)
--   at_floor        mid below 1¢ or from 99¢ up: a contract at the tick floor
--   quote_age_days  how long the quote had been in force at the snapshot
-- Parameters: $series (list of series tickers), $ttms (days before release), $snapshot_hour.
WITH c AS (
    SELECT
        ticker,
        event_ticker,
        series_ticker,
        strike,
        close_time,
        CAST(result = 'yes' AS INTEGER) AS y,
        CAST(max(close_time) OVER (PARTITION BY event_ticker) AT TIME ZONE 'America/New_York' AS DATE)
            AS release_date
    FROM all_markets
    WHERE status = 'finalized' AND series_ticker IN (SELECT unnest($series))
),
grid AS (
    SELECT
        c.*,
        h.ttm,
        ((c.release_date - h.ttm) + to_hours(CAST($snapshot_hour AS BIGINT)))::TIMESTAMP
            AT TIME ZONE 'America/New_York' AS snapshot_ts
    FROM c
    CROSS JOIN (SELECT unnest($ttms) AS ttm) h
)
SELECT
    g.*,
    q.ts AS quote_ts,
    q.bid_close AS bid,
    q.ask_close AS ask,
    q.ask_close - q.bid_close AS spread,
    (q.bid_close + q.ask_close) / 2 AS p,
    epoch(g.snapshot_ts - q.ts) / 86400 AS quote_age_days,
    (q.bid_close + q.ask_close) / 2 < 0.01 OR (q.bid_close + q.ask_close) / 2 >= 0.99 AS at_floor
FROM grid g
ASOF JOIN candles_raw q ON g.ticker = q.ticker AND g.snapshot_ts >= q.ts
WHERE g.snapshot_ts < g.close_time AND q.bid_close IS NOT NULL AND q.ask_close IS NOT NULL
ORDER BY g.series_ticker, g.event_ticker, g.ttm, g.strike;


-- name: trades_by_day
-- Trades in settled contracts of $series per ET calendar day before the release, days
-- 1..$max_days. Used to cut the time-to-release buckets so each holds about the same number
-- of trades. Parameters: $series, $max_days.
WITH c AS (
    SELECT
        ticker,
        CAST(max(close_time) OVER (PARTITION BY event_ticker) AT TIME ZONE 'America/New_York' AS DATE)
            AS release_date
    FROM all_markets
    WHERE status = 'finalized' AND series_ticker IN (SELECT unnest($series))
),
d AS (
    SELECT date_diff('day', CAST(t.created_time AT TIME ZONE 'America/New_York' AS DATE), c.release_date)
               AS days_to_release
    FROM trades_raw t
    JOIN c USING (ticker)
)
SELECT days_to_release, count(*) AS trades
FROM d
WHERE days_to_release BETWEEN 1 AND $max_days
GROUP BY days_to_release
ORDER BY days_to_release;
