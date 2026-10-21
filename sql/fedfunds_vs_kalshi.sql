-- Queries for 03_fedfunds_vs_kalshi.py. Load kalshi_views.sql first (for `markets`)
-- and define a `candles_clean` view over the cleaned 1-minute candles.


-- name: hourly_path
-- The quote in force at the top of every hour, per contract of one meeting, from the contract's
-- first candle to its close. Candles exist only when something changed, so each hour takes the
-- latest candle at or before it. Parameter: $event_ticker.
WITH m AS (
    SELECT ticker, move, move_bps, close_time
    FROM markets
    WHERE event_ticker = $event_ticker
),
hours AS (
    SELECT
        m.ticker,
        unnest(generate_series(
            date_trunc('hour', min(c.ts)) + INTERVAL 1 HOUR,
            any_value(m.close_time),
            INTERVAL 1 HOUR
        )) AS ts
    FROM candles_clean c
    JOIN m USING (ticker)
    GROUP BY m.ticker
)
SELECT
    h.ticker,
    m.move,
    m.move_bps,
    h.ts,
    c.bid_close AS bid,
    c.ask_close AS ask,
    (c.bid_close + c.ask_close) / 2 AS mid,
    c.ask_close - c.bid_close AS spread
FROM hours h
JOIN m USING (ticker)
ASOF JOIN candles_clean c ON h.ticker = c.ticker AND h.ts >= c.ts
ORDER BY m.move_bps, h.ts;


-- name: snapshots
-- The quote in force at $snapshot_hour ET on each (ticker, date) row of a registered
-- `snapshot_grid` table. Rows before a contract's first candle have no quote and drop out.
-- Parameter: $snapshot_hour (hour of day, ET).
WITH g AS (
    SELECT
        ticker,
        date,
        (date + to_hours(CAST($snapshot_hour AS BIGINT)))::TIMESTAMP AT TIME ZONE 'America/New_York' AS snapshot_ts
    FROM snapshot_grid
)
SELECT
    g.ticker,
    g.date,
    c.ts AS quote_ts,
    c.bid_close AS bid,
    c.ask_close AS ask,
    c.ask_close - c.bid_close AS spread,
    (c.bid_close + c.ask_close) / 2 AS kalshi_p
FROM g
ASOF JOIN candles_clean c ON g.ticker = c.ticker AND g.snapshot_ts >= c.ts;


-- name: minute_mids
-- Every candle's bid/ask mid for a list of contracts, for the minute-level reliability diagram.
-- Candles exist only when something changed; the notebook carries each mid forward to every minute.
-- Parameter: $tickers (list of contract tickers).
SELECT ticker, ts, (bid_close + ask_close) / 2 AS mid
FROM candles_clean
WHERE ticker IN (SELECT unnest($tickers))
ORDER BY ticker, ts;
