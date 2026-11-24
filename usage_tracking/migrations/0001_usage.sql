CREATE TABLE usage_events (
  id INTEGER PRIMARY KEY,
  day TEXT NOT NULL,
  tool TEXT NOT NULL CHECK (tool IN ('imc', 'mcd', 'merscope')),
  affiliation TEXT NOT NULL CHECK (affiliation IN ('manchester', 'external')),
  name TEXT,
  name_key TEXT,
  event_id TEXT NOT NULL,
  CHECK ((name IS NULL) = (name_key IS NULL)),
  UNIQUE (day, tool, event_id)
);
CREATE UNIQUE INDEX named_daily_usage ON usage_events(day, tool, affiliation, name_key) WHERE name_key IS NOT NULL;
CREATE INDEX usage_by_day ON usage_events(day);
CREATE VIEW monthly_usage AS
SELECT substr(day, 1, 7) AS month, tool, affiliation,
  count(*) AS usage_days,
  count(DISTINCT name_key) AS named_users,
  sum(name IS NULL) AS anonymous_activity_days
FROM usage_events GROUP BY month, tool, affiliation;
