TRUNCATE TABLE train;
TRUNCATE TABLE client;
TRUNCATE TABLE gas_prices;
TRUNCATE TABLE electricity_prices;

\copy train FROM '/app/enefit/data/raw_data/train.csv' WITH CSV HEADER
\copy client FROM '/app/enefit/data/raw_data/client.csv' WITH CSV HEADER
\copy gas_prices FROM '/app/enefit/data/raw_data/gas_prices.csv' WITH CSV HEADER
\copy electricity_prices FROM '/app/enefit/data/raw_data/electricity_prices.csv' WITH CSV HEADER

