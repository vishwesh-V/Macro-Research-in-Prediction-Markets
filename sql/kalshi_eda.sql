-- Starter EDA queries on the views in kalshi_views.sql (load that file first).
-- Each query is separated by a `-- name:` line so a notebook can run them one by one.


-- name: liquidity_by_month
-- Monthly traded contracts and median spread, to see the May 2025 liquidity break.
SELECT
    date_trunc('month', created_time) AS month,
    sum(contracts) AS contracts,
    count(*) AS trades,
    median(contracts) AS median_trade_size
FROM trades
GROUP BY ALL
ORDER BY month;


-- name: spread_by_move
-- Median spread per outcome bucket in the last 60 days before each meeting
-- (all quotes, including wide ones).
-- Tail buckets with wide spreads are where the favorite-longshot bias would hide.
SELECT
    move,
    move_bps,
    count(*) AS contract_days,
    median(spread) AS median_spread,
    median(mid) AS median_mid
FROM candles_daily
WHERE days_to_meeting BETWEEN 1 AND 60
GROUP BY ALL
ORDER BY move_bps;


-- name: probability_path
-- Daily mid for every bucket of one meeting: the market's probability path.
-- Parameter $event_ticker, e.g. con.execute(sql, {"event_ticker": "KXFEDDECISION-26OCT"}).
SELECT date, days_to_meeting, move, move_bps, bid, ask, mid, spread, volume
FROM candles_daily
WHERE event_ticker = $event_ticker
ORDER BY date, move_bps;


-- name: sum_of_mids
-- The buckets of a meeting should sum to ~1. Large deviations flag stale quotes.
-- `buckets_quoted` counts buckets with a mid.
SELECT
    event_ticker,
    date,
    count(mid) AS buckets_quoted,
    count(*) AS buckets,
    sum(mid) AS sum_mid
FROM candles_daily
WHERE days_to_meeting BETWEEN 1 AND 60
GROUP BY ALL
ORDER BY event_ticker, date;


-- name: calibration_bins
-- Crude calibration: bucket the daily mid into 10 bins, compare with how often
-- the outcome happened. Settled meetings only, t in [-60, -1] days.
-- Days within a meeting are highly correlated, so the counts overstate precision.
SELECT
    floor(d.mid * 10) / 10 AS price_bin,
    count(*) AS contract_days,
    count(DISTINCT d.event_ticker) AS meetings,
    avg(d.mid) AS avg_price,
    avg(CASE WHEN m.result = 'yes' THEN 1 ELSE 0 END) AS hit_rate
FROM candles_daily d
JOIN markets m USING (ticker)
WHERE m.status = 'finalized' AND d.days_to_meeting BETWEEN 1 AND 60 AND d.mid IS NOT NULL
GROUP BY ALL
ORDER BY price_bin;
