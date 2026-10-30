# Troubleshooting browser tools

Start by identifying whether the **application itself** fails to open, or the
application opens but **your dataset** does not. These require different checks.

## The website is blank or does not open

1. Check the address against the [browser tools index](index.md).
2. Reload the page. If it remains blank, try a hard refresh: **Ctrl+Shift+R** on
   Windows/Linux or **Command+Shift+R** on macOS in Chrome/Edge.
3. Try a current desktop Chrome or Edge browser. Note whether the browser shows
   a connection/privacy error, a blank page, or an application message.
4. If permitted on your network, compare with another connection, such as
   mobile data. A site that works there but fails on the institutional network
   may need a network administrator to check filtering or routing.

Do not bypass browser certificate warnings. Report the exact error and address.
Do not clear site storage as a first troubleshooting step: it can remove saved
workspaces and annotations. Export your workspace JSON first if the app is
accessible. A private window can help compare browser state, but it will not
show your normal profile's saved workspaces.

## The file or folder picker is slow

A folder chooser may need to list many files before returning control to the
app, particularly on a network drive. An indeterminate progress indicator does
not necessarily mean the application has stopped.

- Ensure the mapped drive is connected and cloud-synchronised files are
  available locally.
- Select the requested dataset or asset folder, rather than an entire drive or
  a parent containing many unrelated projects.
- In IMC Explorer, prefer **Connect folder** and the primary file/folder
  controls over the compatibility chooser for large network roots.
- In MERSCOPE Explorer, use **Open local / network folder** where available in
  Chrome/Edge; the compatibility picker lists the selected folder's files.
- If access is unreliable, try a local copy that preserves the dataset's
  internal folder structure. Keep the original data unchanged.
- For MERSCOPE, a complete ZIP is another option. An uncompressed ZIP permits
  direct reads; compressed entries may need more memory to decompress.

Use **Cancel** or **Cancel loading** when offered. Reads already issued to the
operating system may finish after cancellation. Repeatedly opening the same
dataset while it is loading is unlikely to speed it up.

## The dataset opens, but images or cells are missing

| Tool | First checks |
| --- | --- |
| IMC Explorer | Match ROI names, mask filenames and image folders exactly. Confirm the ROI/object-ID columns and equal image/mask dimensions. Grey mask objects may have no matching AnnData row. |
| MCD Explorer | Select the intended acquisition and visible marker channels. Check notices for missing pixels or unsupported acquisition data. |
| MERSCOPE Explorer | Choose the correct coordinate system, add the required layers, and check their visibility and order. For cell colours, inspect the table association and exact region/cell-ID match count. |

Start with one visible layer or channel and automatic contrast. Then restore
other layers and filters one at a time. A hidden layer, population filter or
display range can obscure valid data; changing these display controls does
not change the source measurements.

## Expression colours look wrong

- Confirm the selected gene or observation and the source table. Different
  segmentations may be associated with different tables.
- Check the expression source. IMC Explorer offers `X` and available
  `layers`; MERSCOPE's shapes expression controls use `X`. Neither guide
  assumes that these stored values are raw counts or already normalised.
- Check the legend, colour limits, linear/log display choice and population
  filters. Automatic scaling can differ between regions or views.
- In IMC Explorer, image normalisation dictionaries affect TIFF channel
  images, not AnnData expression values.
- In MERSCOPE, gene names match exactly, including case. “Present” in a table
  means available for reading, not necessarily nonzero expression.

Do not interpret an unmatched cell, missing gene or unsupported read as a
biological zero. Use the reported match counts and warnings.

## A large dataset or export runs out of memory

Browser memory includes images, tables, geometry and GPU resources. A tool's
allocation guard is not a guarantee that every device can comfortably display
data of that size.

Reduce the number of panes, visible channels and layers. Use fewer transcript
genes or a smaller embedding context sample, and reduce export resolution.
Close unrelated memory-intensive tabs. For format-specific limits, consult
the relevant [IMC](imc_explorer.md#supported-inputs-and-limits),
[MCD](mcd_explorer.md#limits-and-troubleshooting) or
[MERSCOPE](merscope_explorer.md#supported-inputs-and-limits) guide.

## A workspace is missing or will not reconnect

Saved workspaces are specific to the website address, browser profile and
computer. A save made at one address does not automatically appear at another.
Private browsing or clearing site storage can remove local saves.

- **IMC Explorer:** open the saved workspace and use **Assets & connection**
  to choose its data root. Resolve items marked **Needs attention** using
  **Locate file/folder**, **Edit path**, and **Retry**. Use a full asset check
  after changing files.
- **MERSCOPE Explorer:** reopen the same store. A copied, modified or repackaged
  store, or a different folder-opening method, can be treated as a different
  local workspace. Import an exported workspace JSON after opening the correct
  store to transfer its settings.
- **MCD Explorer:** display settings and export bookmarks are session-only;
  they are not restored after reload.

A workspace JSON contains settings and, where applicable, review annotations;
it is not a backup of the underlying scientific data. It cannot grant file
access on another computer. Keep the original assets and exported settings
together, and check that restored views refer to the intended dataset.

## Report a problem

Use the app's **Report an issue** link where available, or open a
[GitHub issue](https://github.com/dr-michael-haley/imcanalysis/issues). Include:

- Tool name, displayed version, website address, browser/version and OS.
- The steps from opening the app to the failure, including the chosen file
  type and approximate size or number of files.
- The exact error, or whether it is a blank page, stalled file selection,
  missing layer, incorrect display or failed export.
- Whether the same action works in another browser or on another network.
- For a data problem, the relevant format, compression, coordinate-system or
  identity-column details, without attaching a private dataset.

The in-app email link opens your mail application; it does not send a message
or attach data automatically. A minimal non-sensitive example is more useful
than an entire research dataset.
