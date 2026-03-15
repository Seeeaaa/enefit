TRUNCATE TABLE train;
TRUNCATE TABLE client;

\copy train FROM '/app/enefit/data/raw_data/train.csv' WITH CSV HEADER
\copy client FROM '/app/enefit/data/raw_data/client.csv' WITH CSV HEADER
