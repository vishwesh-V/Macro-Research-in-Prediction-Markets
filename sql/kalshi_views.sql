-- Views over the Kalshi parquet files written by 00_fetch_data_kalshi.py.
-- Paths are relative to the repo root, so run from there.
--
-- From Python / marimo:
--     import duckdb
--     con = duckdb.connect()
--     con.execute(open("sql/kalshi_views.sql").read())
--     con.sql("SELECT * FROM meeting_summary").df()
--
-- The views are cheap: DuckDB reads only the columns and row groups a query needs.


-- One row per contract. `move_bps` normalizes Kalshi's inconsistent bucket codes:
-- "cut >25bps" appears as C26, C>25, C24 and TC25; "hold" as H0 with four
-- different labels. Open-ended buckets get the next 25bp step (cut >25 -> -50).
CREATE OR REPLACE VIEW markets AS
SELECT
    ticker,
    event_ticker,
    series_ticker,
    CAST(close_time AT TIME ZONE 'America/New_York' AS DATE) AS meeting_date,
    bucket,
    CASE
        WHEN bucket IN ('C26', 'C>25', 'C24', 'TC25') THEN -50
        WHEN bucket = 'C25'                           THEN -25
        WHEN bucket = 'H0'                            THEN 0
        WHEN bucket = 'H25'                           THEN 25
        WHEN bucket IN ('H26', 'H>25', 'H50')         THEN 50
        WHEN bucket = 'TH50'                          THEN 75
    END AS move_bps,
    CASE
        WHEN bucket IN ('C26', 'C>25', 'C24', 'TC25') THEN 'Cut >25'
        WHEN bucket = 'C25'                           THEN 'Cut 25'
        WHEN bucket = 'H0'                            THEN 'Hold'
        WHEN bucket = 'H25'                           THEN 'Hike 25'
        WHEN bucket IN ('H26', 'H>25')                THEN 'Hike >25'
        WHEN bucket = 'H50'                           THEN 'Hike 50'
        WHEN bucket = 'TH50'                          THEN 'Hike >50'
    END AS move,
    outcome AS kalshi_label,
    status,
    result,
    open_time,
    close_time,
    volume
FROM 'data/kalshi_markets.parquet';


-- Every trade, with the contract's meeting and move attached.
CREATE OR REPLACE VIEW trades AS
SELECT
    t.ticker,
    m.event_ticker,
    m.meeting_date,
    m.move_bps,
    m.move,
    t.created_time,
    t.yes_price,
    t.count AS contracts,
    t.taker_side,
    t.is_block_trade,
    t.trade_id
FROM 'data/kalshi_trades.parquet' t
JOIN markets m USING (ticker);


-- 1-minute candles. Sparse: a row exists only when the quote or a trade changed.
-- `mid` and `spread` use the closing best bid/ask of the minute.
CREATE OR REPLACE VIEW candles AS
SELECT
    c.ticker,
    m.event_ticker,
    m.meeting_date,
    m.move_bps,
    m.move,
    c.ts,
    c.bid_close AS bid,
    c.ask_close AS ask,
    (c.bid_close + c.ask_close) / 2 AS mid,
    c.ask_close - c.bid_close AS spread,
    c.price_close AS last_trade,
    c.volume,
    c.open_interest,
    date_diff('day', CAST(c.ts AT TIME ZONE 'America/New_York' AS DATE), m.meeting_date) AS days_to_meeting
FROM 'data/kalshi_candles_1m.parquet' c
JOIN markets m USING (ticker);


-- One row per contract per calendar day (ET), from its first candle to the
-- meeting (or today): the quote in force at the end of each day, carried forward
-- over days with no candle, plus that day's volume. The usual input for
-- calibration over t in [-60, -1] days.
-- `mid` is NULL when the spread is wider than 20c: a 0c/100c book says nothing.
CREATE OR REPLACE VIEW candles_daily AS
WITH days AS (
    SELECT
        ticker,
        unnest(generate_series(
            min(CAST(ts AT TIME ZONE 'America/New_York' AS DATE)),
            least(any_value(meeting_date), current_date),
            INTERVAL 1 DAY
        ))::DATE AS date
    FROM candles
    GROUP BY ticker
),
eod AS (  -- end of each ET day, as a UTC timestamp comparable with candles.ts
    SELECT ticker, date, (date + INTERVAL 1 DAY)::TIMESTAMP AT TIME ZONE 'America/New_York' AS day_end
    FROM days
),
daily_volume AS (
    SELECT ticker, CAST(ts AT TIME ZONE 'America/New_York' AS DATE) AS date, sum(volume) AS volume
    FROM candles
    GROUP BY ALL
)
SELECT
    c.ticker,
    c.event_ticker,
    c.meeting_date,
    c.move_bps,
    c.move,
    e.date,
    date_diff('day', e.date, c.meeting_date) AS days_to_meeting,
    c.bid,
    c.ask,
    CASE WHEN c.spread <= 0.20 THEN c.mid END AS mid,
    c.spread,
    coalesce(v.volume, 0) AS volume,
    c.open_interest,
    c.ts AS quote_ts  -- when this quote was last updated
FROM eod e
ASOF JOIN candles c ON e.ticker = c.ticker AND e.day_end > c.ts
LEFT JOIN daily_volume v ON v.ticker = e.ticker AND v.date = e.date;


-- One row per meeting: what happened and how much traded.
CREATE OR REPLACE VIEW meeting_summary AS
SELECT
    event_ticker,
    meeting_date,
    count(*) AS contracts,
    any_value(move) FILTER (WHERE result = 'yes') AS decision,
    sum(volume) AS volume,
    min(open_time) AS first_listed,
    bool_and(status = 'finalized') AS settled
FROM markets
GROUP BY ALL
ORDER BY meeting_date;
