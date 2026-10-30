# MCD Explorer

A small companion to SBT Tissue Explorer with browser-local computation. Open an MCD directly;
no TIFF conversion, Python analysis service, or data upload is involved.

## Run

From this folder, with Node.js 22 or later:

```shell
node server.mjs
```

Open http://127.0.0.1:8878 in desktop Chrome or Edge. An optional first argument
selects a different port: `node server.mjs 8879`. The server serves only app
assets on loopback; MCD files are read in the browser. No npm installation or
build is required. The static index and src files can also be served over HTTP
or HTTPS. Opening index.html via file:// does not support the worker.

## Explore

- Open or drop one `.mcd` file. Slides and acquisitions come from its metadata.
- Choose a region in the acquisition list or click its outline on the overview.
- In **02 Acquisitions → Backdrop**, choose the panorama, the embedded whole-slide
  photo, or coordinates only. The whole-slide photo includes all available ROI
  outlines on the current slide. Unavailable images are marked in the selector.
  Imported photographs (such as `PXL_….jpg`) are detected in the MCD's panorama
  records, even when stored internally as PNG. Each imported photo is listed
  by name; the photograph with the largest slide coverage is offered first.
  The separate slide overview remains available too.
  Scroll over the overview to zoom, drag to pan, and click an ROI to open it.
  The +/− and Fit controls also work without a mouse wheel. With the overview
  focused, use +/−, arrow keys, and F/Home. Zoom is retained when switching
  ROIs on the same backdrop; choosing another backdrop resets the view.
- Mix up to six channels; choose pure red, green, blue, cyan, yellow, magenta,
  or white. Each component is exactly 0 or 255. Use the arrows beside each
  marker selector to cycle forwards/backwards, wrapping at the ends while
  keeping the slot's colour and visibility. Each new marker gets auto contrast.
  Viridis, Plasma, Magma and Inferno are also available per channel, with a
  colour-scale preview. Maps blend additively; show one channel for an unmixed
  scale. The standard maps retain their dark low-end colour, including at zero.
- Adjust raw-count Min/Max with two handles on a shared track or number fields.
  Both use 0.1-count steps; arrow keys move a focused handle by 0.1 and Page
  Up/Down by 1. Bounds stay at least 0.1 apart. Auto uses zero/minimum and the
  sampled 99.5th percentile, rounded outward to 0.1 counts.
- **Advanced** contains Display transform (linear by default, plus arcsinh,
  log1p and square root) and Channel order. Menus and marker arrows default to
  alphabetical order; MCD order preserves the original metal-tag order and
  binary channel indices. Changing order keeps the current channel selections.
  Arcsinh defaults to cofactor 5. Display transforms do not alter raw values.
- **Histogram** lists the channels currently shown. Select any subset and click
  Generate histogram. A local worker bins every finite measured pixel into 128
  shared raw-count bins; measured zeros are included and missing pixels excluded.
  Curves show pixels per bin, with optional log frequency. No histograms run
  automatically. Changing acquisition, visible channels or the histogram subset
  clears the previous result; contrast and colour edits need no regeneration.
- Drag to pan, scroll to zoom, and use Fit view, 1:1, or double-click to refit.
- Click an image pixel to inspect the selected channels' original values.
- Choose embedded panorama, slide, or before/after-ablation images when present.
- Inspect acquisition metadata in **06 Metadata**.
- The IMC view includes a scale bar calibrated at **1 image pixel = 1 µm**.
  Its length remains correct as you zoom. Optical photographs are uncalibrated.
- In **07 Image Export**, switch to Figure to preview an export. Set the aspect
  ratio or lock the frame to a physical width and height, then pan to frame it.
  Returning to Viewer restores your previous framing. Add a title and channel
  legend, choose a colour scheme and text sizes, and set an automatic or manual
  scale bar. Export PNG, JPEG, or SVG at a chosen pixel width; SVG embeds the
  composite image with editable vector labels, legend and scale bar layers.
  File names can include a prefix, acquisition, title, timestamp and selected
  acquisition metadata. Optional export bookmarks restore the acquisition,
  channels, display settings and framing within the current file and tab.
  Bookmarks clear when the file is closed or the tab is reloaded.
- Try the explicitly labelled synthetic demo to exercise the complete reader.

## First-version limits

Float32 acquisitions only. Maximum 16,777,216 image pixels, 16,384 pixels per
side, and a 256 MiB decoded-data budget per selection. One acquisition is held
at a time. Changing to uncached channels rereads that acquisition in 4 MiB
chunks; the whole MCD is never loaded into memory. Closing the tab releases
the file. Display settings are not persisted between sessions.

The viewer reports incomplete records, repeated or invalid pixel coordinates,
and missing measurements. Missing acquisition pixels use a checker pattern.
IMC pixels are calibrated to 1 µm; optical photographs have no scale bar. Optical
images support browser-decodable JPEG, PNG, and BMP encodings. Panorama outlines
use the MCD's recorded corners, including for imported whole-slide photographs.
The separate slide overview uses its recorded WidthUm and HeightUm, with the
slide origin at the bottom left. An externally cropped
or rotated phone photo may not align exactly with those coordinates. If the
positioning metadata is missing, the photo is shown without ROI outlines.
Unsupported optical images leave a coordinate map available. Instrument/version
coverage needs testing with real MCD files.

All code and fonts are local. The app's content security policy disallows
network connections. The Node process serves app assets only and accepts no
uploads. See THIRD-PARTY-NOTICES.txt for readimc attribution.

Image exports are limited to 8,192 pixels in width, 16,384 pixels per side and
32 million pixels overall. SVG exports have a 128 MiB size limit. Figure controls
adapt Tissue Explorer's framing, overlay, filename and bookmark features to a
single image; no cell annotations or multiple image panes are included.

## Cloudflare hosting

The production target is `https://mcd.spatialbiology.app/`, served by the separate
`sbt-mcd-explorer` Worker with static assets only. `wrangler.jsonc` uses the same
Cloudflare account as Tissue Explorer. Build and verify from this folder:

```shell
node --test tests/*.test.mjs
node build.mjs
wrangler whoami
wrangler deploy
```

Authenticate to the account in the configuration before deploying. The build
contains only `index.html`, browser code and logo, license notices, and `_headers`.
The headers preserve the local viewer's security policy, including
`connect-src 'none'`. No file upload endpoint, dataset, test harness or local
server is published. Selecting a local MCD on the hosted site reads it on the
user's computer. Users must select their file again when changing origins.

For dashboard deployment, open the existing `sbt-mcd-explorer` Worker and choose
**New deployment**. Upload a ZIP containing the **contents** of `dist/` at its
root (including `_headers`), then deploy. Keep the assets directory `/`, the
default HTML handling, and not-found handling `none`. Do not create another
Worker or upload the project folder when publishing an update.

## Checks

```shell
node --test tests/*.test.mjs
```

Tests exercise binary offsets, chunk boundaries, pixel placement, incomplete
data, memory guards, embedded image headers, and metadata discovery using
small synthetic files. They do not compare against Python.
