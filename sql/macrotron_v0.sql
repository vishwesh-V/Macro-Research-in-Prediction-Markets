-- Queries for 05_macrotron_v0.py. Load kalshi_views.sql first (for `markets` and `trades`), then the view
-- block of kalshi_eda_macro.sql (for `macro_markets` and `macro_daily`), and register a `snapshot_grid`
-- table of (ticker, date) rows as in 03.


-- name: macro_ladders
-- The ladder each CPI and payrolls release faced the evening before: every strike's end-of-day mid on the day
-- before the release (NULL when the strike had no quote yet), and how the strike settled. A strike reads
-- "above k", so the mids trace P(print > k) and the settled YES/NO split brackets the print.
SELECT
    m.series_ticker,
    m.event_ticker,
    m.release_date,
    m.status,
    m.strike,
    m.result,
    d.mid,
    d.spread
FROM macro_markets m
LEFT JOIN macro_daily d ON d.ticker = m.ticker AND d.date = m.release_date - 1
WHERE m.series_ticker IN ('KXCPI', 'KXPAYROLLS')
  AND m.strike IS NOT NULL
ORDER BY m.series_ticker, m.release_date, m.strike;


-- name: last_trades
-- Hours from each contract's last trade to $snapshot_hour ET on each (ticker, date) row of `snapshot_grid`.
-- NULL when the contract had not traded yet. Parameter: $snapshot_hour (hour of day, ET).
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
    date_diff('second', t.created_time, g.snapshot_ts) / 3600.0 AS hours_since_trade
FROM g
ASOF LEFT JOIN trades t ON g.ticker = t.ticker AND g.snapshot_ts >= t.created_time;
