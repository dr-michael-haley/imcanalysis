"""Shared editable SVG export used by figures and legacy backgating."""
import re
import matplotlib as mpl

def _svg_font_family(font_family):
    """Validate and quote one family for CSS without emitting a fallback list."""
    if not isinstance(font_family, str) or not font_family.strip() or ',' in font_family:
        raise ValueError('font_family must be a single nonempty font name, e.g. Arial.')
    return "'" + font_family.strip().replace('\\', '\\\\').replace("'", "\\'") + "'"


def _set_svg_font_family(root, font_family):
    """Normalize font declarations in new and legacy SVGs, preserving live text.

    Handle both separate CSS font-family declarations and older Matplotlib font
    shorthand while retaining size, weight, style, line height and positioning.
    Text content and path/image data are never rewritten.
    """
    family = _svg_font_family(font_family)

    def normalize(style):
        style = re.sub(r'(?<![\w-])font-family\s*:[^;}]*',
                       lambda _: f'font-family: {family}', style, flags=re.IGNORECASE)
        return re.sub(
            r'(?<![\w-])(font\s*:\s*[^;{}]*?\b[\d.]+(?:px|pt|em|rem|%|pc|in|cm|mm)'
            r'(?:\s*/\s*[\d.]+(?:px|pt|em|rem|%)?)?\s+)[^;}]+',
            lambda m: m[1] + family, style, flags=re.IGNORECASE)

    for element in root.iter():
        if 'style' in element.attrib:
            element.set('style', normalize(element.get('style')))
        if 'font-family' in element.attrib:
            element.set('font-family', font_family.strip())
        if element.tag == '{http://www.w3.org/2000/svg}style' and element.text:
            element.text = normalize(element.text)
        if element.tag in ('{http://www.w3.org/2000/svg}text', '{http://www.w3.org/2000/svg}tspan'):
            # Explicitly cover text that previously inherited a generic family.
            element.set('font-family', font_family.strip())


def _save_population_overlay_svg(fig, output_path, *, layers=None, font_family='Arial', group_prefixes=None, **save_kwargs):
    """Save named, editable SVG groups with optional Inkscape layer metadata.

    Illustrator can edit the groups and vector/text objects; its importer may
    display the groups beneath a single native layer.
    """
    from io import BytesIO
    from xml.etree import ElementTree as ET

    _svg_font_family(font_family)
    svg_ns = 'http://www.w3.org/2000/svg'
    inkscape_ns = 'http://www.inkscape.org/namespaces/inkscape'
    ET.register_namespace('', svg_ns)
    ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')
    ET.register_namespace('inkscape', inkscape_ns)
    with BytesIO() as buffer:
        with mpl.rc_context({'svg.fonttype': 'none', 'image.composite_image': False,
                             'svg.image_inline': True}):
            fig.savefig(buffer, format='svg', **save_kwargs)
        root = ET.fromstring(buffer.getvalue())
    _set_svg_font_family(root, font_family)

    layers = dict(layers) if layers is not None else {
        'source_image': 'Source image',
        'cell_outlines': 'Cell outlines',
        'cell_centers': 'Cell centers',
        'scale_bar': 'Scale bar',
        'scale_bar_text': 'Scale bar text',
        'marker_legend': 'Marker legend',
        'population_label': 'Population label',
        'primary_panel_title': 'Primary panel title',
    }
    # Comparison image pixels, text, legends and cell paths stay independently
    # editable, including when these SVGs are subsequently assembled into grids.
    import re
    for element in root.iter():
        match = re.fullmatch(r'(comparison_\d+)_(image|title|legend|missing|cell_centers|cell_outline_.+)', element.get('id', ''))
        if match:
            prefix, component = match.groups()
            component = 'cell_outlines' if component.startswith('cell_outline_') else component
            layers.setdefault(f'{prefix}_{component}', f'{prefix.replace("_", " ").title()}: {component.replace("_", " ")}')
    # Keep backend transforms, clipping, and drawing order intact. Gather the
    # individual cell paths and the two scale-bar rectangles at their parent.
    for parent in list(root.iter()):
        groups = {}
        for child in list(parent):
            gid = child.get('id', '')
            key = ('cell_outlines' if gid.startswith('cell_outline_') else
                   'scale_bar' if gid in ('scale_bar_fill', 'scale_bar_outline') else gid)
            if re.fullmatch(r'comparison_\d+_cell_outline_.+', gid):
                key = gid.split('_cell_outline_')[0] + '_cell_outlines'
            for prefix, group_key in (group_prefixes or {}).items():
                if gid.startswith(prefix):
                    key = group_key
                    break
            if key not in layers:
                continue
            if key not in groups:
                group = ET.Element(f'{{{svg_ns}}}g', {
                    'id': key,
                    f'{{{inkscape_ns}}}groupmode': 'layer',
                    f'{{{inkscape_ns}}}label': layers[key],
                })
                parent.insert(list(parent).index(child), group)
                groups[key] = group
            if gid == key:
                child.set('id', f'{key}_content')
            parent.remove(child)
            groups[key].append(child)
    ET.ElementTree(root).write(output_path, encoding='utf-8', xml_declaration=True)


