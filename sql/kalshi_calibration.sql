-- Queries for 02_calibration_kalshi.py. Load kalshi_views.sql first (for `markets`)
-- and define a `candles_clean` view over the cleaned 1-minute candles.


-- name: forecasts
-- One row per settled contract and horizon: the quote in force at $snapshot_hour ET,
-- `ttm` days before the meeting, and whether the contract settled YES (`y`).
-- Contracts not yet listed at a horizon have no quote and drop out.
--   p               raw bid/ask midpoint
--   q               p divided by the sum of p over the meeting's buckets at that snapshot,
--                   so the buckets sum to 1 (NULL unless every bucket is quoted)
--   at_floor        mid below 1¢ or from 99¢ up: a contract at the tick floor
--                   (0¢ bid / 1¢ ask, or no ask), not a priced forecast
--   quote_age_days  how long the quote had been in force at the snapshot
-- Parameters: $ttms (list of days to meeting), $snapshot_hour (hour of day, ET).
WITH grid AS (
    SELECT
        m.ticker,
        m.event_ticker,
        m.meeting_date,
        m.move,
        m.move_bps,
        CAST(m.result = 'yes' AS INTEGER) AS y,
        h.ttm,
        ((m.meeting_date - h.ttm) + to_hours(CAST($snapshot_hour AS BIGINT)))::TIMESTAMP
            AT TIME ZONE 'America/New_York' AS snapshot_ts,
        count(*) OVER (PARTITION BY m.event_ticker, h.ttm) AS buckets
    FROM markets m
    CROSS JOIN (SELECT unnest($ttms) AS ttm) h
    WHERE m.status = 'finalized'
),
quoted AS (
    SELECT
        g.*,
        c.ts AS quote_ts,
        c.bid_close AS bid,
        c.ask_close AS ask,
        c.ask_close - c.bid_close AS spread,
        (c.bid_close + c.ask_close) / 2 AS p,
        epoch(g.snapshot_ts - c.ts) / 86400 AS quote_age_days
    FROM grid g
    ASOF JOIN candles_clean c ON g.ticker = c.ticker AND g.snapshot_ts >= c.ts
)
SELECT
    * EXCLUDE (buckets),
    sum(p) OVER w AS sum_p,
    CASE WHEN count(*) OVER w = buckets THEN p / sum(p) OVER w END AS q,
    p < 0.01 OR p >= 0.99 AS at_floor
FROM quoted
WINDOW w AS (PARTITION BY event_ticker, ttm)
ORDER BY event_ticker, ttm, move_bps;


-- name: minute_quotes
-- Every quote of a settled contract from midnight ET $days_before days before the meeting until
-- the contract closes, with how long it was in force (`seconds`). Candles exist only when something
-- changed, so a quote lasts until the contract's next candle or the close; the quote in force at
-- the window start counts from the window start. Parameter: $days_before.
WITH m AS (
    SELECT
        ticker,
        event_ticker,
        CAST(result = 'yes' AS INTEGER) AS y,
        close_time,
        (meeting_date - $days_before)::TIMESTAMP AT TIME ZONE 'America/New_York' AS window_start
    FROM markets
    WHERE status = 'finalized'
),
q AS (
    SELECT
        m.ticker,
        m.event_ticker,
        m.y,
        (c.bid_close + c.ask_close) / 2 AS p,
        c.ask_close - c.bid_close AS spread,
        greatest(c.ts, m.window_start) AS t0,
        least(coalesce(lead(c.ts) OVER (PARTITION BY c.ticker ORDER BY c.ts), m.close_time), m.close_time) AS t1
    FROM candles_clean c
    JOIN m USING (ticker)
    WHERE c.ts < m.close_time
)
SELECT event_ticker, ticker, y, p, spread, sum(epoch(t1 - t0)) AS seconds
FROM q
WHERE t1 > t0
GROUP BY ALL
ORDER BY event_ticker, ticker, p;
