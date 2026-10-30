# SpatialData Explorer

A read-only, browser-local SpatialData viewer with the dark sidebar and mint
controls used by SBT Tissue Explorer and MCD Explorer. Open a local SpatialData
`.zarr` directory or ZIP, choose a coordinate system, and combine images,
segmentation labels, shapes and points. Mix up to six image channels, colour
cells by an associated table column, click features to inspect annotations,
save/restore view settings and export the canvas as PNG.

The app has no data server or Python runtime. A static host serves HTML, CSS,
JavaScript, workers and WASM. The browser reads selected local files on demand;
there is no upload endpoint. Folder selection enumerates file paths, so very
large stores with many small files can take time to open. An uncompressed ZIP
reduces that enumeration overhead and permits direct range reads; compressed
entries may require decompression of the whole entry.

## Repository boundary

`imcanalysis/spatialdata_viewer` owns branding, layout, file selection, controls,
view files, build configuration and deployment. The
[SpatialData.js fork](https://github.com/dr-michael-haley/SpatialData.js) owns
reusable loaders, table joins, rendering and the embedding API. Both execute in
the browser. There is no backend service to deploy.

`engine.lock.json` pins the fork by commit. Release builds use its frozen pnpm
lockfile; they do not consume a moving branch or require an npm release of the
fork. The generated `build-info.json` records the engine revision and whether
local development changes were present. Once the engine API stabilises, versioned
package releases can replace the source checkout without moving the app.

## Build and run

Use Node 24.14.1 and pnpm 11.20.0 (the fork's packageManager version). From this
directory:

```shell
npm run prepare:engine
pnpm --dir .engine/SpatialData.js install --frozen-lockfile
npm test
npm run build
npm run preview
```

Open **http://127.0.0.1:8879** in desktop Chrome or Edge. The preview server
serves only `dist/` and accepts GET/HEAD. User datasets are opened through the
browser file picker, never through the server. No Python or npm installation is
needed on users' machines when the app is hosted on HTTPS. `file://` is not
supported because module workers and WASM need an HTTP/HTTPS origin.

To develop against the adjacent fork on this computer, use PowerShell:

```powershell
$env:SBT_SPATIALDATA_ROOT = 'D:/Github_repos/SpatialData.js'
pnpm --dir $env:SBT_SPATIALDATA_ROOT install --frozen-lockfile
npm run build
npm run preview
```

`npm run dev` rebuilds the engine packages once, then starts the app dev server.
After changing engine source, restart it to rebuild the engine. Clear
`SBT_SPATIALDATA_ROOT` before making a release build. The prepared checkout must
match the pin and have no tracked modifications; development overrides are
explicitly recorded in build provenance. Build scripts resolve the fork's
locked tools and public package entry points, so the app needs no second copy
of React or deck.gl.

## Deploy

Publish the **contents of `dist/`** to a static host. Keep workers and WASM at
their generated paths, serve `.wasm` as `application/wasm`, and preserve
`_headers` on Cloudflare. All runtime dependencies are bundled locally; there
are no CDN scripts, remote datasets or fonts. The same-origin connection policy
allows loading the app's own workers/WASM. The codec bindings need
`script-src 'unsafe-eval'` during initialization; removing it breaks Blosc
decoding. The app does not evaluate imported view files as code.

`wrangler.jsonc` is a separate static-assets deployment named
`sbt-spatialdata-explorer`. After authenticating to the intended Cloudflare
account, `wrangler deploy` from this directory publishes the built assets.
No account or custom domain is preselected, and this does not update the Tissue
Explorer or MCD Explorer sites. A dedicated subdomain such as
`spatialdata.spatialbiology.app` can be attached when the deployment is ready.

The manual **Build SpatialData Explorer** GitHub Actions workflow builds the
pinned engine and uploads a static-site artifact. It does not deploy a site.

## Validation

```shell
npm test
python tests/make_fixture.py
npm run test:browser
```

Fixture generation needs the scientific SBT environment, including spatialdata,
AnnData and GeoPandas. It writes a fresh synthetic store and archive under the
ignored `fixtures/` directory and records their names in `fixture.json`. The
fixture uses Zarr v2, two coordinate systems, multiscale Blosc-compressed images
and masks, Parquet shapes/points, and shuffled table rows with repeated IDs
across regions. A process-local adapter handles the installed Dask/OME-Zarr
writer API mismatch without changing the environment.

Browser tests use desktop Chrome by default (`SBT_BROWSER_CHANNEL` can select
another installed Playwright channel). They check loading, channel controls,
region-aware annotation picking, saved views, PNG output, invalid-file recovery,
same-origin GET-only requests and unchanged input-file hashes.

## Current limits

- Initial 2D viewer. Image z/t slice controls are available when axes exist;
  there is no 3D scene navigation, linked UMAP, expression plots or editing.
- The first associated table supplies colouring and tooltips when several
  tables annotate an element. View files match element names and coordinate
  systems, not a content checksum; reopen the correct dataset before importing.
- Chunk decoding has a 256 MiB cache ceiling, not a whole-browser memory limit.
  Tables, geometry, GPU textures and large unchunked arrays need additional
  memory. Large point sets follow the engine's limits and report notices.
- Format/codec support follows the pinned SpatialData.js engine. Automated
  browser coverage currently uses the synthetic Zarr v2 fixture; large real
  stores, Zarr v3 and other codec combinations still need representative testing.
- PNG exports screen-resolution canvas pixels. Scientific figure composition,
  editable legends, calibration controls and high-resolution export are not
  implemented here.

Bundled dependencies retain their licences in `THIRD_PARTY_LICENSES.txt`.
