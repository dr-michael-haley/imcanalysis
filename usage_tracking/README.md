# Browser tool usage

A dependency-free first-visit overlay and a Cloudflare Worker with a D1 database
and public [explorer usage dashboard](https://usage.spatialbiology.app/).
The explorers remain static applications; scientific files never enter the
recording service. There is no login or verified identity.

## Behaviour and data

- Ask for a name **or** an anonymous choice, plus University of Manchester / External.
- Remember preferences in localStorage on the current origin. The three subdomains
  ask separately. Private browsing, clearing storage and another browser/device
  can cause the prompt to reappear. Blocked storage falls back to memory.
- **Not now** and Escape dismiss the prompt without sending a record; ask again
  on a later page load. **Usage preferences** reopens it at any time.
- Record a tool opening after the user has chosen, at most once per UTC day in
  that browser. Returning to a visible tab on a later day can record another day.
  This measures openings, not completed analyses, time spent or login events.
- Daily reporting sends `tool`, `name` (null when anonymous), `affiliation`, and `eventId`.
  The server supplies the UTC date. No dataset content, filenames, URL parameters,
  IP addresses or user agents are stored in D1. Cloudflare still handles ordinary
  network requests; anonymous does not mean invisible to the hosting provider.
- The daily event UUID supports retries. It changes with the day/tool and is not
  an enduring anonymous user identifier. Anonymous activity days are **not unique
  people**. Clearing storage or changing choices may add anonymous records.
- Names are self-reported. Normalized, case-folded names deduplicate per day,
  tool and affiliation. Different people with the same name can be conflated.
- Changing preferences affects future records, not past records. Switching to
  anonymous removes the prior name from the component's local storage.
- The two collection endpoints are write-only, validate origins/tool pairs and bounded JSON,
  uses prepared SQL, and never logs submitted payloads. CORS is not authentication:
  non-browser clients can forge an origin and submit records. Treat reports as
  indicative usage evidence, not an audit or access-control system.
- Failed reporting has a five-second timeout and never blocks use. Daily records retry on a
  later visit, network recovery or tab visibility; do not silently queue old days.

### Action counts

An independent, initially unchecked preference enables action counts. Returning
visitors can review it on their next visit. Each successful AnnData, MCD or
SpatialData open sends one `dataset_opened`. The MCD built-in example sends
`demo_opened` instead. Each generated figure download sends one
`figure_exported`. Reopening the same dataset counts again. Cancelled pickers,
unreadable metadata and failed exports do not count. An accepted dataset may
still have a later rendering error.

`POST /actions` accepts only `tool`, `affiliation`, `action` and a fresh random
`eventId`. It stores the server's UTC day. The receipt makes duplicate delivery
idempotent. Actions have no user, visit or dataset identifier, even when a
person has supplied a name for daily reporting. They are best effort and are
not queued while offline. The client can disable them at any time.

## Add to another static tool

Copy `client/usage.js` into the tool's published assets, and add:

```html
<script defer src="./usage.js"
  data-usage-tool="imc"
  data-usage-endpoint="https://usage.spatialbiology.app/record"
  data-usage-hosts="imc.spatialbiology.app"></script>
```

Change the tool ID and host for your tool. `data-usage-hosts` is a comma-separated
hostname allowlist: the integrations intentionally disable collection on localhost
and unofficial mirrors. Omit it to enable the component on any host, if desired.
Add the exact HTTPS origin and tool pair to `TOOL_ORIGINS` in `wrangler.jsonc`.
For a new tool ID, add a migration updating the SQL tool constraint as well.
Allow `https://usage.spatialbiology.app` in the page's `connect-src` CSP. The script
is same-origin; the scoped styles require the explorers' existing inline-style
permission. The UI uses a shadow root to avoid changing the host app's styles.

Vite applications can use `usageComponentPlugin()` from `vite-plugin.mjs`; it
serves the shared source during development and emits `usage.js` in production.
IMC and MERSCOPE use that plugin. MCD's explicit build copies the same file,
and its development server serves only that specific shared file.
Call `window.sbtRecordAction?.('dataset_opened')` at the point where the app
accepts a dataset and `window.sbtRecordAction?.('figure_exported')` when it
generates the download. Keep filenames and other event properties out of calls.

## Cloudflare setup

Use Node 24+ and Wrangler (validated with 4.152.0). From this directory:

```sh
wrangler whoami
# If the saved login lacks D1 access, reauthorize including d1:write.
wrangler login --scopes user:read account:read workers:write workers_routes:write workers_scripts:write zone:read ssl_certs:write d1:write
wrangler d1 create sbt-tool-usage --location weur
```

For a fresh account, change `account_id`, routes and allowed origins first. Put the
returned database UUID in `d1_databases[0].database_id` in `wrangler.jsonc`.
For this account, reuse the configured database; do not recreate it.

```sh
wrangler d1 migrations apply sbt-tool-usage --remote
wrangler deploy --dry-run
wrangler deploy
```

The dashboard and its aggregate API are public. Wrangler uses
your Cloudflare credentials. `/health` is a liveness check only, not a DB check.
Worker observability is disabled so the service does not retain request logs.
Stay on the Free plan; the service uses only Workers requests and D1 storage/SQL.
If limits are reached, recording can fail while the explorers remain usable.

## Read reports

The dashboard compares apps and plots daily activity. Filter by date, app or
affiliation, then download the app comparison or daily totals as CSV.
`GET /api/v1/usage` accepts optional `from`, `to` (inclusive UTC dates, up to 366
days), `tool` (`all`, `imc`, `mcd`, `merscope`) and `affiliation` (`all`,
`manchester`, `external`). It defaults to the last 30 UTC days. The response
contains filters, the action coverage start day, aggregate participation, totals, app rows and daily
rows. Null denotes unavailable coverage or an inapplicable metric. Zero means
there are no records for an available measure. Historical activity days remain
available. Counts may be cached for 60 seconds.

Only grouped counts leave the database through the reporting API. Named daily
records remain in the existing restricted D1 table. The public website, API and
CSV omit names, per-person labels, event IDs and individual event records.
Participation counts distinct normalized supplied names across the entire
selection and activity days reported anonymously. Filtering to one app gives
the participation counts for that explorer.

In Cloudflare, go to **Storage & databases → D1 → sbt-tool-usage → Console**.
Run the following SQL, or use the equivalent CLI:

```sql
SELECT * FROM monthly_usage ORDER BY month DESC, tool, affiliation;
SELECT day, tool, affiliation, name FROM usage_events ORDER BY day DESC;
```

```sh
wrangler d1 execute sbt-tool-usage --remote --command="SELECT * FROM monthly_usage ORDER BY month DESC, tool, affiliation" --json
```

`usage_days` counts named daily records plus anonymous daily activity records.
`named_users` counts distinct normalized supplied names in that month/tool/group.
`anonymous_activity_days` counts unnamed records, never unique anonymous users.
No retention purge is enabled by default. Review stored names periodically;
authorized operators can delete records through the D1 console. Cloudflare backup
retention is separate from live-table deletion. Do not publish raw names in reports.

## Verification and rollout

```sh
node --test tests/worker.test.mjs
node --test tests/browser.test.mjs
```

Worker tests exercise the actual migration in SQLite, validation, deduplication,
CORS and DB failures. Browser tests use the repository's Playwright and installed
Chrome (`SBT_BROWSER_CHANNEL` overrides the channel), intercept all recording
requests, and exercise the three real explorer builds. Build all three explorers
first. No test names are sent to production. Set `SBT_USAGE_STAGED=1` to test the
ignored `deployment/<tool>/updated` snapshots instead of the normal build outputs.

For the initial rollout, `scripts/stage-live.mjs` checks every built asset against
the live site before staging a usage-only change. It refuses mismatched builds.
It preserves the original assets and Wrangler configuration under ignored
`deployment/<tool>/original*` for rollback. Once staged, refresh the `updated/usage.js`
copies if the component changes. Deploy each `updated.json` config only after
the recording service works. To roll back, deploy that tool's `original.json`.
Future normal app builds include the integration automatically. The staging script
is for the initial live retrofit, not a routine build or a way to publish unrelated
local changes. Browser tests can reuse the retained snapshots.

`node scripts/smoke-live.mjs` explicitly tests the production sites and writes
three anonymous test records. It saves their exact IDs and a `smoke-cleanup.sql`
file under ignored `deployment/`. After verifying those IDs in D1, remove only
those records with `wrangler d1 execute sbt-tool-usage --remote --file=deployment/smoke-cleanup.sql`.

## Initial deployment — 6 November 2026

Published and verified on all three custom domains. Database:
`sbt-tool-usage` (`d8f82400-5775-4ea4-a875-359bbad4d4e2`, Western Europe location hint).

| Worker | Deployed version |
| --- | --- |
| sbt-tool-usage | 5fdcd41a-ee09-40a3-a92d-31f214e1ebff |
| sbt-imc-explorer | d33e6d63-893c-436e-9de8-2826fb018638 |
| sbt-mcd-explorer | 40eddb64-089c-418a-8661-2dda3596a308 |
| sbt-merscope-explorer | 9cf2b0bd-fac5-42af-9cfa-2d6d49b993ba |

Validation: all three production builds passed; 9 focused usage tests and 29 MCD
unit tests passed; repository validation and whitespace checks passed. Live browser
submissions returned 204, produced exactly one anonymous record per tool, and
remembered the choice on reload. Test records were removed after verification.
The production retrofit preserved the existing deployed explorer bundles.

## Dashboard and action reporting release, 19 November 2026

All four Workers use the existing spatialbiology.app account and D1 database.
Migration `0002_actions.sql` adds anonymous action receipts and the first day
of action coverage. No historical dataset or figure counts were inferred.

| Worker | Production version |
| --- | --- |
| sbt-tool-usage | 8172196a-b9dd-41c9-96dd-ac1390574ff3 |
| sbt-imc-explorer | d6baaf94-2087-40ea-81ba-4b11a48890c5 |
| sbt-mcd-explorer | ade0d5ec-6161-4639-a633-d33ffde65d4d |
| sbt-merscope-explorer | 299531bd-de2c-4bc6-8877-2d4c510f4945 |

On the deployment workstation, set
`$env:XDG_CONFIG_HOME = Join-Path $env:LOCALAPPDATA 'SpatialBiologyToolkit/cloudflare'`
in PowerShell before invoking Wrangler. Verify the account ID with
`wrangler whoami` against `account_id` in the chosen configuration.

IMC and MCD shipped through their normal builds. For MERSCOPE this release
applied the two compiled action hooks to the verified live 0.5.6 bundle, keeping
a pending local layout change out of production. The source hooks are also in
`spatialdata_viewer/src/main.jsx` for subsequent normal builds.

Run `node scripts/check-dashboard.mjs` from this directory for a read-only check
of the live dashboard and CSV downloads at desktop and mobile sizes. Set
`SBT_USAGE_LIVE_UI=1` for `node --test tests/browser.test.mjs` to exercise
the deployed explorers with synthetic datasets and intercepted usage requests.
