# SpatialData Explorer

SpatialData Explorer opens local SpatialData Zarr folders and ZIP archives
entirely in a desktop browser. It combines multichannel images, segmentation,
points, shapes and associated cell annotations in a 2D view. Display settings
can be saved as JSON and the visible canvas exported as PNG.

The app uses SBT's IMC Explorer styling. Its reusable loading and rendering
engine comes from the pinned SBT fork of SpatialData.js. Files remain on the
user's computer; hosting requires only static HTML, JavaScript, CSS and WASM.

Build and run from `spatialdata_viewer/` with Node 24.14.1 and pnpm 11.20.0:

```shell
npm run prepare:engine
pnpm --dir .engine/SpatialData.js install --frozen-lockfile
npm run build
npm run preview
```

Open `http://127.0.0.1:8879` in Chrome or Edge, choose the `.zarr` folder itself
or its ZIP, then select a coordinate system and layers. The preview process
serves application assets only. When hosted on HTTPS, users need just a browser.

This initial version does not include 3D navigation, linked UMAP plots or data
editing. Large datasets remain subject to browser and GPU memory limits.
Automated browser tests cover a synthetic Zarr v2 store; representative real
datasets and other formats/codecs need further validation.

See the [application README](https://github.com/dr-michael-haley/imcanalysis/tree/main/spatialdata_viewer)
for development, deployment, checks and limitations.
