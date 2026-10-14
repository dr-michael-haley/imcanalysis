"""Apply NapariSBT's shared workflow palette to its Sphinx guide."""
from pathlib import Path

from docutils import nodes
from SpatialBiologyToolkit.napari_sbt.help.presentation import ACCENTS, SECTION_ACCENTS


def style_help_sections(app, doctree, docname):
    if not docname.startswith("napari_sbt/"):
        return
    topic = app.env.metadata.get(docname, {}).get("napari_help_topic", "")
    colours = SECTION_ACCENTS.get(topic, {})
    for section in doctree.findall(nodes.section):
        section["classes"].append("napari-guide-section")
        title = next((child for child in section if isinstance(child, nodes.title)), None)
        if title is not None and title.astext() in colours:
            section["classes"].append(f"napari-accent-{colours[title.astext()]}")


def write_palette(app, exception):
    if exception is not None or app.builder.format != "html":
        return
    output = Path(app.outdir) / "_static" / "napari-help-colours.css"
    output.parent.mkdir(parents=True, exist_ok=True)
    rules = []
    for name, (border, _title_border, background, foreground) in ACCENTS.items():
        selectors = ", ".join(
            f".rst-content .napari-accent-{name} > h{level}"
            for level in (1, 2, 3, 4)
        )
        rules.append(
            f"{selectors} {{ border-left: 5px solid {border}; "
            f"background: {background}; color: {foreground}; "
            "padding: .55rem .8rem; border-radius: 5px; }"
        )
    output.write_text("\n".join(rules) + "\n", encoding="utf-8")


def setup(app):
    app.add_css_file("napari-help-colours.css")
    app.connect("doctree-resolved", style_help_sections)
    app.connect("build-finished", write_palette)
    return {"version": "1", "parallel_read_safe": True, "parallel_write_safe": True}
