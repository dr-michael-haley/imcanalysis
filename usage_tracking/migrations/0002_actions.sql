-- Action receipts are independent of the daily, optionally named records.
CREATE TABLE usage_actions (
  event_id TEXT PRIMARY KEY NOT NULL,
  day TEXT NOT NULL,
  tool TEXT NOT NULL CHECK (tool IN ('imc', 'mcd', 'merscope')),
  affiliation TEXT NOT NULL CHECK (affiliation IN ('manchester', 'external')),
  action TEXT NOT NULL CHECK (action IN ('dataset_opened', 'figure_exported')
    OR (tool = 'mcd' AND action = 'demo_opened'))
) WITHOUT ROWID;
CREATE INDEX actions_by_day ON usage_actions(day, tool, affiliation, action);

CREATE TABLE usage_coverage (
  metric TEXT PRIMARY KEY,
  start_day TEXT NOT NULL
);
INSERT INTO usage_coverage VALUES ('actions', date('now'));
