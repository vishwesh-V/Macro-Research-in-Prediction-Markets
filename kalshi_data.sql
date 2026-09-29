-- Minimal test: can we pull Kalshi market data?
-- Requires DuckDB (https://duckdb.org). Run with:  duckdb < kalshi_data.sql
-- Public market-data endpoints need no API key.

INSTALL httpfs;
LOAD httpfs;
-- Kalshi returns 404 on HEAD requests; this makes DuckDB skip the HEAD and just GET
SET force_download = true;

-- CPI markets (series KXCPI): one row per strike/threshold contract
SELECT
    m.ticker,
    m.event_ticker,
    m.yes_sub_title,
    m.status,
    m.yes_bid_dollars,
    m.yes_ask_dollars,
    m.last_price_dollars,
    m.volume_fp,
    m.open_interest_fp,
    m.close_time
FROM (
    SELECT unnest(markets) AS m
    FROM read_json_auto('https://api.elections.kalshi.com/trade-api/v2/markets?series_ticker=KXCPI&limit=20')
)
ORDER BY m.event_ticker, m.ticker;
