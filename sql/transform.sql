DROP TABLE IF EXISTS merged;

CREATE TABLE merged AS
WITH electricity_prepared AS (
    SELECT
        origin_date + INTERVAL '2 days' AS datetime,
        euros_per_mwh,
        data_block_id
    FROM electricity_prices
)
SELECT
    t.datetime,
    t.county,
    t.product_type,
    t.is_business,
    t.is_consumption,
    t.target,
    t.data_block_id,
    t.prediction_unit_id,
    c.eic_count,
    c.installed_capacity,
    gp.lowest_price_per_mwh,
    gp.highest_price_per_mwh,
    ep.euros_per_mwh,
    t.row_id
FROM train AS t
LEFT JOIN client AS c
    USING (data_block_id, county, product_type, is_business)
LEFT JOIN gas_prices AS gp
    ON t.data_block_id = gp.data_block_id
LEFT JOIN electricity_prepared AS ep
    ON t.datetime = ep.datetime
    AND t.data_block_id = ep.data_block_id;

