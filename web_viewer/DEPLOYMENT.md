# IMC Explorer deployment

Renamed from Tissue Explorer on 30 October 2026. The application is version
0.6.3. The source directory and `sbt gui web` launcher remain unchanged.

## Production targets

| Purpose | Worker | Configuration | Active version |
| --- | --- | --- | --- |
| [IMC Explorer](https://imc.spatialbiology.app/) | `sbt-imc-explorer` | `wrangler.jsonc` | `88d91177-0f89-43d6-9285-a96a2704e5c4` |
| [Previous address](https://explorer.spatialbiology.app/) | `sbt-tissue-explorer` | `wrangler.legacy.jsonc` | `0850ca2b-85ec-49fb-8c60-3d764b5f25f7` |

Wrangler confirmed both versions serving 100% of traffic. The app was deployed
at 21:16 UTC; the redirect at 21:19 UTC. The redirect has no static assets: GET
and HEAD return 308 to the fixed HTTPS IMC origin, preserving path and query.
Other methods return 405. Its cache lifetime is five minutes.

Only generated application assets are published from
`../SpatialBiologyToolkit/browser_viewer/static/`. No scientific datasets or
local review evidence are included. The existing authenticated Cloudflare
account and security headers are retained.

## Release order and verification

1. Published documentation commit `475f255ecd53320fac9f07835c864b9d44f5be1c`.
   Read the Docs build [35534496](https://app.readthedocs.org/projects/imcanalysis/builds/35534496/)
   succeeded. The old guide redirects to `browser_tools/imc_explorer.html`,
   preserving query and fragment; this was verified in the browser.
2. Built the application and passed 109 JavaScript unit tests, including the
   redirect tests and workspace compatibility tests; three Python launcher
   tests also passed. A synthetic AnnData file loaded locally, autosaved and
   remained in the workspace list after reload.
3. Deployed the new app. Its workers.dev address rendered version 0.6.3 with
   the new name and guide links. Cloudflare's API confirmed the custom domain
   enabled on the correct service with a certificate ID. Public Google DNS
   resolved the hostname to Cloudflare addresses.
4. Updated the resource hub (`0955f64c-5cf4-468c-bf2d-9de69736156c`) and MCD
   Explorer (`64327c5a-9e23-466a-a018-0e209ba0fb55`). The live MCD app's new
   cross-link was checked. The hub's old `#tissue-explorer` bookmark remains an
   alias for its new `#imc-explorer` resource entry.
5. Deployed the legacy redirect. A live HEAD request to the old workers.dev
   address returned 308 with the expected encoded path and query destination.

Custom-domain HTTPS could not be fully verified from the development network:
the new hostname remained unresolved locally while public DNS resolved it,
and a direct request using that public address encountered a connection reset.
Other spatialbiology.app custom domains had the same existing network problem.
No certificate validation or network protections were bypassed. Application
rendering was verified at the deployed workers.dev address; custom-domain
end-to-end reachability remains a verification limitation.

MERSCOPE Explorer required no runtime change for this rename. Its README
reference was updated, without publishing unrelated in-progress source changes.

## Future updates

Build and verify the app before `wrangler deploy` from this directory. Deploy
the redirect separately only when needed:

```shell
wrangler deploy --config wrangler.legacy.jsonc
wrangler deployments status
wrangler deployments status --config wrangler.legacy.jsonc
```

The new origin starts with a separate browser-local workspace library. No
automatic migration is provided. Previously exported JSON can still be imported
and its original files reconnected. The `sbt-tissue-explorer-workspace` format
identifier, schema version 13 and historical IndexedDB name are intentionally
unchanged for compatibility; they are not product display names.

## Rollback

To restore the former app at the old address, first inspect the deployment
history and then roll back the legacy Worker to the last application version:

```shell
wrangler rollback cb4c2645-2c00-4578-8e8d-a52beb4c18bc --config wrangler.legacy.jsonc
wrangler deployments status --config wrangler.legacy.jsonc
```

This changes the old Worker deployment, not the new IMC Worker. Cached redirects
may persist for five minutes. For a complete naming rollback, also revert the
rename commits and rebuild the hub and affected app; do not delete either domain.
The prior hub version was `08c0a510-155f-48a5-b721-dce826d96882`; the prior MCD
version was `59d9ea53-4768-4ea1-9d6d-5b873da57886`. Rollbacks do not migrate
browser storage between origins.
