"""
Create an HTML report for RAPIDS ParameterScan QC outputs.

Run from a dataset folder by default:

    python Misc_scripts/rapids_parameter_scan_report.py

The default input is ParameterScan and the default output is
rapids_parameter_scan_report.html inside that folder.

Plots are embedded in the HTML by default. Use --link-files to keep the HTML
small and link to the plot files instead.
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import mimetypes
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".svg", ".webp", ".pdf"}


@dataclass
class PlotPair:
    leiden_slug: str
    umap_path: Optional[Path]
    matrixplot_path: Optional[Path]


@dataclass
class ScanEntry:
    label: str
    scan_dir: Path
    summary: Dict[str, Any]
    pairs: List[PlotPair]


def clean_slug(value: Any) -> str:
    """Mirror the pipeline's simple cleanstring behavior for filename matching."""
    text = str(value)
    text = re.sub(r"[^\w]+", "_", text)
    text = re.sub(r"^_+|_+$", "", text)
    text = re.sub(r"_+", "_", text)
    return text


def natural_sort_key(value: Any) -> List[Any]:
    """Sort labels containing numbers in a human-readable order."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(value))
    ]


def is_supported_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def load_summary_rows(scan_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load rapids_parameter_scan_summary.csv when present."""
    summary_path = scan_dir / "rapids_parameter_scan_summary.csv"
    if not summary_path.exists():
        return {}

    rows: Dict[str, Dict[str, Any]] = {}
    with summary_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = row.get("label") or ""
            if not label:
                continue
            rows[label] = row
    return rows


def discover_scan_dirs(scan_dir: Path) -> List[Path]:
    """Find one-level scan result directories."""
    if not scan_dir.exists():
        raise FileNotFoundError(f"ParameterScan directory does not exist: {scan_dir}")

    candidates = [path for path in scan_dir.iterdir() if path.is_dir()]
    scan_dirs = []
    for candidate in candidates:
        has_umap = any(is_supported_image(path) for path in candidate.glob("umap_*"))
        has_matrixplot = any(
            is_supported_image(path)
            for path in (candidate / "Matrixplots").glob("Matrixplot_*")
        )
        if has_umap or has_matrixplot:
            scan_dirs.append(candidate)

    return sorted(scan_dirs, key=lambda path: natural_sort_key(path.name))


def leiden_slug_from_matrixplot(path: Path) -> Optional[str]:
    """Extract the cleaned Leiden key from Matrixplot_<leiden>_vmax.<ext>."""
    stem = path.stem
    prefix = "Matrixplot_"
    suffix = "_vmax"
    if not stem.startswith(prefix) or not stem.endswith(suffix):
        return None
    leiden_slug = stem[len(prefix) : -len(suffix)]
    return leiden_slug or None


def leiden_slug_from_umap(path: Path, batch_slug: str) -> Optional[str]:
    """Extract the cleaned Leiden key from a slide-vs-Leiden UMAP filename."""
    stem = path.stem
    marker = f"{batch_slug}_vs_"
    if marker in stem:
        leiden_slug = stem.split(marker, 1)[1]
        return leiden_slug or None

    match = re.search(r"(leiden_.+)$", stem)
    return match.group(1) if match else None


def find_plot_pairs(scan_dir: Path, batch_key: str) -> List[PlotPair]:
    """Pair UMAPs and MatrixPlots by cleaned Leiden key."""
    batch_slug = clean_slug(batch_key) or "slide"
    matrixplots = [
        path
        for path in (scan_dir / "Matrixplots").glob("Matrixplot_*_vmax.*")
        if is_supported_image(path)
    ]
    umaps = [
        path
        for path in scan_dir.glob("umap_*")
        if is_supported_image(path)
    ]

    matrix_by_slug: Dict[str, Path] = {}
    for path in sorted(matrixplots, key=lambda item: natural_sort_key(item.name)):
        leiden_slug = leiden_slug_from_matrixplot(path)
        if leiden_slug and leiden_slug not in matrix_by_slug:
            matrix_by_slug[leiden_slug] = path

    umap_by_slug: Dict[str, Path] = {}
    for path in sorted(umaps, key=lambda item: natural_sort_key(item.name)):
        leiden_slug = leiden_slug_from_umap(path, batch_slug)
        if leiden_slug and leiden_slug not in umap_by_slug:
            umap_by_slug[leiden_slug] = path

    leiden_slugs = sorted(
        set(matrix_by_slug) | set(umap_by_slug),
        key=natural_sort_key,
    )
    return [
        PlotPair(
            leiden_slug=leiden_slug,
            umap_path=umap_by_slug.get(leiden_slug),
            matrixplot_path=matrix_by_slug.get(leiden_slug),
        )
        for leiden_slug in leiden_slugs
    ]


def parse_json_field(value: Any) -> Any:
    """Parse JSON summary fields when possible."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def format_summary_items(summary: Dict[str, Any]) -> List[tuple[str, str]]:
    """Return compact metadata rows for a scan card."""
    fields = [
        "method",
        "n_cells",
        "n_markers",
        "n_pcs",
        "n_pcs_neighbors",
        "n_neighbors",
        "umap_min_dist",
        "run_harmony",
        "harmony_flavor",
        "matrixplot_count",
    ]
    items: List[tuple[str, str]] = []
    for field in fields:
        value = summary.get(field)
        if value not in (None, ""):
            items.append((field, str(value)))

    overrides = parse_json_field(summary.get("overrides"))
    if isinstance(overrides, dict) and overrides:
        overrides_text = ", ".join(f"{key}={value}" for key, value in overrides.items())
        items.insert(0, ("overrides", overrides_text))
    return items


def rel_link(path: Optional[Path], output_path: Path) -> Optional[str]:
    """Create a POSIX-style relative link from the output HTML file."""
    if path is None:
        return None
    rel_path = os.path.relpath(path.resolve(), start=output_path.parent.resolve())
    return Path(rel_path).as_posix()


def guess_media_type(path: Path) -> str:
    """Return a browser-friendly media type for an image or PDF path."""
    if path.suffix.lower() == ".svg":
        return "image/svg+xml"
    media_type, _ = mimetypes.guess_type(path.name)
    return media_type or "application/octet-stream"


def media_source(path: Path, output_path: Path, embed_media: bool) -> str:
    """Return either a data URI or a relative file link for a media file."""
    if embed_media:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{guess_media_type(path)};base64,{encoded}"

    href = rel_link(path, output_path)
    assert href is not None
    return href


def render_media(
    path: Optional[Path],
    output_path: Path,
    alt: str,
    *,
    embed_media: bool,
) -> str:
    """Render an image or PDF block."""
    if path is None:
        return '<div class="missing">Missing file</div>'

    source = media_source(path, output_path, embed_media)
    escaped_source = html.escape(source)
    escaped_alt = html.escape(alt)

    if path.suffix.lower() == ".pdf":
        fallback = (
            f'<p><a href="{escaped_source}">Open PDF: {escaped_alt}</a></p>'
            if not embed_media
            else f"<p>Embedded PDF preview is unavailable: {escaped_alt}</p>"
        )
        return (
            f'<object class="plot pdf-plot" data="{escaped_source}" type="application/pdf">'
            f"{fallback}"
            f"</object>"
        )

    img = f'<img class="plot" src="{escaped_source}" alt="{escaped_alt}" loading="lazy">'
    if embed_media:
        return img

    return (
        f'<a href="{escaped_source}">'
        f"{img}"
        f"</a>"
    )


def render_metadata_table(summary: Dict[str, Any]) -> str:
    """Render scan metadata as a small table."""
    items = format_summary_items(summary)
    if not items:
        return '<p class="muted">No summary metadata found for this scan.</p>'

    rows = "\n".join(
        "<tr>"
        f"<th>{html.escape(key)}</th>"
        f"<td>{html.escape(value)}</td>"
        "</tr>"
        for key, value in items
    )
    return f'<table class="meta"><tbody>{rows}</tbody></table>'


def render_scan(
    entry: ScanEntry,
    output_path: Path,
    batch_key: str,
    *,
    embed_media: bool,
) -> str:
    """Render one scan card."""
    title = html.escape(entry.label)
    rel_dir = html.escape(rel_link(entry.scan_dir, output_path) or str(entry.scan_dir))
    metadata = render_metadata_table(entry.summary)

    if not entry.pairs:
        pairs_html = (
            '<div class="empty">'
            "No paired slide-vs-Leiden UMAPs or MatrixPlots were found in this scan."
            "</div>"
        )
    else:
        pair_blocks: List[str] = []
        for pair in entry.pairs:
            leiden_label = html.escape(pair.leiden_slug)
            umap = render_media(
                pair.umap_path,
                output_path,
                alt=f"{entry.label} {pair.leiden_slug} {batch_key} vs Leiden UMAP",
                embed_media=embed_media,
            )
            matrixplot = render_media(
                pair.matrixplot_path,
                output_path,
                alt=f"{entry.label} {pair.leiden_slug} matrixplot",
                embed_media=embed_media,
            )
            pair_blocks.append(
                f"""
                <section class="pair">
                  <h3>{leiden_label}</h3>
                  <div class="plot-grid">
                    <figure>
                      <figcaption>UMAP: {html.escape(batch_key)} vs {leiden_label}</figcaption>
                      {umap}
                    </figure>
                    <figure>
                      <figcaption>MatrixPlot: {leiden_label}</figcaption>
                      {matrixplot}
                    </figure>
                  </div>
                </section>
                """
            )
        pairs_html = "\n".join(pair_blocks)

    return f"""
    <article class="scan-card" id="{html.escape(clean_slug(entry.label))}">
      <header>
        <h2>{title}</h2>
        <p class="muted">{rel_dir}</p>
      </header>
      {metadata}
      {pairs_html}
    </article>
    """


def build_report(
    *,
    scan_dir: Path,
    output_path: Path,
    batch_key: str,
    title: str,
    embed_media: bool,
) -> str:
    """Build the complete HTML report."""
    summary_rows = load_summary_rows(scan_dir)
    scan_dirs = discover_scan_dirs(scan_dir)
    entries = [
        ScanEntry(
            label=scan_path.name,
            scan_dir=scan_path,
            summary=summary_rows.get(scan_path.name, {}),
            pairs=find_plot_pairs(scan_path, batch_key=batch_key),
        )
        for scan_path in scan_dirs
    ]

    generated = datetime.now().isoformat(timespec="seconds")
    scan_cards = "\n".join(
        render_scan(entry, output_path, batch_key, embed_media=embed_media)
        for entry in entries
    )
    if not scan_cards:
        scan_cards = '<div class="empty">No scan result folders were found.</div>'

    nav_items = "\n".join(
        f'<a href="#{html.escape(clean_slug(entry.label))}">{html.escape(entry.label)}</a>'
        for entry in entries
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #17211b;
      --muted: #627069;
      --line: #d7ded9;
      --paper: #f7f4ed;
      --card: #ffffff;
      --accent: #0f6b58;
    }}
    body {{
      margin: 0;
      background: var(--paper);
      color: var(--ink);
      font-family: "Segoe UI", "Aptos", sans-serif;
      line-height: 1.45;
    }}
    main {{
      max-width: 1500px;
      margin: 0 auto;
      padding: 28px;
    }}
    h1, h2, h3 {{
      line-height: 1.1;
    }}
    h1 {{
      font-size: clamp(2rem, 4vw, 4rem);
      margin: 0 0 8px;
    }}
    .lede {{
      color: var(--muted);
      margin: 0 0 22px;
    }}
    .nav {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 22px 0;
    }}
    .nav a {{
      background: #e5eee9;
      border: 1px solid var(--line);
      border-radius: 999px;
      color: var(--accent);
      padding: 6px 10px;
      text-decoration: none;
      font-size: 0.9rem;
    }}
    .scan-card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: 0 12px 40px rgba(23, 33, 27, 0.08);
      margin: 26px 0;
      padding: 22px;
    }}
    .scan-card h2 {{
      font-size: 1.8rem;
      margin: 0;
    }}
    .muted {{
      color: var(--muted);
      font-size: 0.95rem;
    }}
    .meta {{
      border-collapse: collapse;
      margin: 16px 0 22px;
      width: 100%;
    }}
    .meta th,
    .meta td {{
      border-bottom: 1px solid var(--line);
      padding: 7px 10px;
      text-align: left;
      vertical-align: top;
    }}
    .meta th {{
      color: var(--muted);
      font-weight: 600;
      width: 180px;
    }}
    .pair {{
      border-top: 1px solid var(--line);
      padding-top: 18px;
      margin-top: 18px;
    }}
    .pair h3 {{
      margin: 0 0 12px;
      color: var(--accent);
    }}
    .plot-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
      align-items: start;
    }}
    figure {{
      margin: 0;
      background: #fbfaf7;
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px;
    }}
    figcaption {{
      color: var(--muted);
      font-size: 0.9rem;
      margin-bottom: 10px;
    }}
    .plot {{
      display: block;
      width: 100%;
      max-height: 860px;
      object-fit: contain;
    }}
    .pdf-plot {{
      min-height: 720px;
    }}
    .missing,
    .empty {{
      border: 1px dashed #b9c4bd;
      border-radius: 12px;
      color: var(--muted);
      padding: 18px;
      text-align: center;
    }}
    @media (max-width: 900px) {{
      main {{
        padding: 16px;
      }}
      .plot-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>{html.escape(title)}</h1>
    <p class="lede">
      Source: {html.escape(str(scan_dir))}<br>
      Batch key for UMAP matching: {html.escape(batch_key)}<br>
      Media mode: {"embedded in this HTML file" if embed_media else "linked from QC files"}<br>
      Generated: {html.escape(generated)}<br>
      Scans found: {len(entries)}
    </p>
    <nav class="nav">{nav_items}</nav>
    {scan_cards}
  </main>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an HTML report for RAPIDS ParameterScan QC outputs.",
    )
    parser.add_argument(
        "--scan-dir",
        default="ParameterScan",
        help="Directory containing RAPIDS parameter scan subfolders.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output HTML path. Defaults to <scan-dir>/rapids_parameter_scan_report.html.",
    )
    parser.add_argument(
        "--batch-key",
        default="slide",
        help="Batch obs key used in UMAP filenames, default: slide.",
    )
    parser.add_argument(
        "--title",
        default="RAPIDS Parameter Scan Report",
        help="HTML report title.",
    )
    parser.add_argument(
        "--link-files",
        action="store_true",
        help=(
            "Link image/PDF files instead of embedding them. This keeps the HTML "
            "smaller, but the report is no longer self-contained."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scan_dir = Path(args.scan_dir)
    output_path = Path(args.output) if args.output else scan_dir / "rapids_parameter_scan_report.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = build_report(
        scan_dir=scan_dir,
        output_path=output_path,
        batch_key=args.batch_key,
        title=args.title,
        embed_media=not args.link_files,
    )
    output_path.write_text(report, encoding="utf-8")
    print(f"Wrote RAPIDS ParameterScan report to {output_path}")


if __name__ == "__main__":
    main()
