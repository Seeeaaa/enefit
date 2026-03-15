CREATE TABLE IF NOT EXISTS train (
    county INTEGER,
    is_business BOOLEAN,
    product_type INTEGER,
    target DOUBLE PRECISION,
    is_consumption BOOLEAN,
    datetime TIMESTAMP,
    data_block_id INTEGER,
    row_id BIGINT PRIMARY KEY,
    prediction_unit_id INTEGER
);

CREATE TABLE IF NOT EXISTS client (
    product_type INTEGER,
    county INTEGER,
    eic_count INTEGER,
    installed_capacity DOUBLE PRECISION,
    is_business BOOLEAN,
    date DATE,
    data_block_id INTEGER
);
