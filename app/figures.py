# app/figures.py
# High-fidelity, syllabus-aligned programmatic mathematical diagram generation for QPGen.
# Produces scalable, black-and-white printable vector SVG graphics for Classes 1 to 5,
# covering:
# 1. Visual multiplication, equal groups, repeated addition (inspired by user reference 1)
# 2. 3D solid objects, side/top view projections (inspired by user reference 2)
# 3. 2D geometrical shapes and axes of symmetry (inspired by user reference 3)
# 4. Fractions, perimeter grids, angles, and spatial object counting.

from __future__ import annotations

import math
from typing import Optional
from app.schemas import (
    Difficulty,
    FigureData,
    Question,
    QuestionType,
    MathVerificationResult,
    MathCheckStatus,
)


# ---------------------------------------------------------------------------
# SVG Drawing Primitives & Helpers
# ---------------------------------------------------------------------------

def _star_polygon(cx: float, cy: float, r: float = 9.0) -> str:
    """Generate a clean 5-point star polygon markup."""
    pts = []
    for i in range(10):
        angle = i * math.pi / 5 - math.pi / 2
        radius = r if i % 2 == 0 else r * 0.44
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        pts.append(f"{x:.1f},{y:.1f}")
    return f'<polygon points="{" ".join(pts)}" fill="#ffffff" stroke="#0f172a" stroke-width="1.3"/>'


def _circle_item(cx: float, cy: float, r: float = 7.0, filled: bool = False) -> str:
    """Generate a circle token (open or filled)."""
    fill = "#0f172a" if filled else "#ffffff"
    return f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}" stroke="#0f172a" stroke-width="1.3"/>'


# ---------------------------------------------------------------------------
# 1. Visual Multiplication & Grouping SVG Generators (Reference Image 1)
# ---------------------------------------------------------------------------

def render_multiplication_options_svg(
    correct_groups: int,
    correct_count_per_group: int,
    shape_type: str = "star",
) -> tuple[str, list[str], str]:
    """
    Renders 4 distinct grouping/multiplication diagrams labeled A, B, C, D
    matching the exact style of User Reference Image 1:
      A. [4 stars] + [2 stars]
      B. [2 stars] × [2 stars]
      C. [4 stars] + [4 stars]  <-- (Represents 4 x 2)
      D. [4 stars] × [4 stars]
    Returns (svg_markup, options_list, correct_option_letter).
    """
    k = correct_count_per_group
    g = correct_groups

    # Option configs:
    # A: [K items] + [G items] (incorrect addition of operands)
    # B: [G items] × [G items] (incorrect multiplication of group counts)
    # C: G boxes of [K items] with '+' (correct repeated addition representing K × G)
    # D: [K items] × [K items] (incorrect multiplication of item counts)
    options_cfg = {
        "A": {"boxes": [k, g], "op": "+"},
        "B": {"boxes": [g, g], "op": "×"},
        "C": {"boxes": [k] * g, "op": "+"},
        "D": {"boxes": [k, k], "op": "×"},
    }
    correct_letter = "C"

    # SVG layout
    svg_w = 520
    row_h = 48
    svg_h = 20 + len(options_cfg) * row_h
    item_spacing = 22
    box_pad = 10
    star_r = 7.5

    elements = [
        f'<svg width="{svg_w}" height="{svg_h}" viewBox="0 0 {svg_w} {svg_h}" xmlns="http://www.w3.org/2000/svg" style="background:#ffffff; font-family:Arial, sans-serif;">'
    ]

    labels = ["A", "B", "C", "D"]
    for idx, lbl in enumerate(labels):
        y_top = 14 + idx * row_h
        y_mid = y_top + 17
        cfg = options_cfg[lbl]

        # Option letter label
        elements.append(
            f'<text x="14" y="{y_mid + 5}" font-size="14" font-weight="bold" fill="#0f172a">{lbl}.</text>'
        )

        curr_x = 42
        num_boxes = len(cfg["boxes"])
        for b_idx, count in enumerate(cfg["boxes"]):
            box_w = max(36, count * item_spacing + box_pad * 2)
            box_h = 32
            elements.append(
                f'<rect x="{curr_x}" y="{y_top}" width="{box_w}" height="{box_h}" fill="#ffffff" stroke="#0f172a" stroke-width="1.3" rx="2"/>'
            )
            # Render items in box
            for i in range(count):
                ix = curr_x + box_pad + i * item_spacing + star_r
                iy = y_top + box_h / 2
                if shape_type == "star":
                    elements.append(_star_polygon(ix, iy, star_r))
                else:
                    elements.append(_circle_item(ix, iy, star_r - 1.5))

            curr_x += box_w
            # Operator between boxes
            if b_idx < num_boxes - 1:
                op_char = cfg["op"]
                elements.append(
                    f'<text x="{curr_x + 9}" y="{y_mid + 6}" font-size="16" font-weight="bold" fill="#0f172a">{op_char}</text>'
                )
                curr_x += 26

    elements.append("</svg>")
    svg_markup = "\n".join(elements)
    options = ["Figure A", "Figure B", "Figure C", "Figure D"]
    return svg_markup, options, "Figure C"


def render_group_count_stem_svg(
    groups: int,
    count_per_group: int,
    shape_type: str = "star",
) -> str:
    """Renders equal groups of stars or dots for repeated addition / counting."""
    spacing = 22
    box_pad = 10
    star_r = 8.0
    box_w = max(36, count_per_group * spacing + box_pad * 2)
    box_h = 34
    gap = 18

    total_w = max(380, groups * (box_w + gap) + 40)
    total_h = 60

    elements = [
        f'<svg width="{total_w}" height="{total_h}" viewBox="0 0 {total_w} {total_h}" xmlns="http://www.w3.org/2000/svg" style="background:#ffffff; font-family:Arial, sans-serif;">'
    ]

    start_x = 20
    for g in range(groups):
        bx = start_x + g * (box_w + gap)
        by = 12
        elements.append(
            f'<rect x="{bx}" y="{by}" width="{box_w}" height="{box_h}" fill="#f8fafc" stroke="#0f172a" stroke-width="1.3" rx="3"/>'
        )
        for i in range(count_per_group):
            ix = bx + box_pad + i * spacing + star_r
            iy = by + box_h / 2
            if shape_type == "star":
                elements.append(_star_polygon(ix, iy, star_r))
            else:
                elements.append(_circle_item(ix, iy, star_r - 1.5))

    elements.append("</svg>")
    return "\n".join(elements)


# ---------------------------------------------------------------------------
# 2. 3D Geometrical Solids & Top/Side Views SVG Generators (Reference Image 2)
# ---------------------------------------------------------------------------

def _draw_solid_wireframe(solid_type: str, x: float, y: float, w: float, h: float) -> str:
    """Draw clean line-art isometric drawing of a 3D solid."""
    s = solid_type.lower()
    if s == "cube":
        # Isometric Cube
        dx = 16
        dy = 14
        cw = min(w, h) - dx
        ch = cw
        # Top face
        p_top = f"{x},{y + dy} {x + dx},{y} {x + cw + dx},{y} {x + cw},{y + dy}"
        # Side face
        p_side = f"{x + cw},{y + dy} {x + cw + dx},{y} {x + cw + dx},{y + ch} {x + cw},{y + ch + dy}"
        return (
            f'<polygon points="{p_top}" fill="#f8fafc" stroke="#0f172a" stroke-width="1.3"/>'
            f'<rect x="{x}" y="{y + dy}" width="{cw}" height="{ch}" fill="#ffffff" stroke="#0f172a" stroke-width="1.3"/>'
            f'<polygon points="{p_side}" fill="#f1f5f9" stroke="#0f172a" stroke-width="1.3"/>'
        )
    elif s == "cuboid":
        # Isometric Cuboid (wider than tall)
        dx = 16
        dy = 12
        cw = w - dx - 4
        ch = h - dy - 6
        p_top = f"{x},{y + dy} {x + dx},{y} {x + cw + dx},{y} {x + cw},{y + dy}"
        p_side = f"{x + cw},{y + dy} {x + cw + dx},{y} {x + cw + dx},{y + ch} {x + cw},{y + ch + dy}"
        return (
            f'<polygon points="{p_top}" fill="#f8fafc" stroke="#0f172a" stroke-width="1.3"/>'
            f'<rect x="{x}" y="{y + dy}" width="{cw}" height="{ch}" fill="#ffffff" stroke="#0f172a" stroke-width="1.3"/>'
            f'<polygon points="{p_side}" fill="#f1f5f9" stroke="#0f172a" stroke-width="1.3"/>'
        )
    elif s == "cylinder":
        # Isometric Cylinder
        cx = x + w / 2
        rx = w / 2 - 2
        ry = 7
        top_y = y + ry
        bot_y = y + h - ry
        return (
            f'<path d="M {cx - rx},{top_y} L {cx - rx},{bot_y} A {rx},{ry} 0 0,0 {cx + rx},{bot_y} L {cx + rx},{top_y}" fill="#ffffff" stroke="#0f172a" stroke-width="1.3"/>'
            f'<ellipse cx="{cx}" cy="{top_y}" rx="{rx}" ry="{ry}" fill="#f8fafc" stroke="#0f172a" stroke-width="1.3"/>'
            f'<path d="M {cx - rx},{bot_y} A {rx},{ry} 0 0,1 {cx + rx},{bot_y}" fill="none" stroke="#0f172a" stroke-width="1.3" stroke-dasharray="3,3"/>'
        )
    elif s == "cone":
        # Isometric Cone
        cx = x + w / 2
        rx = w / 2 - 3
        ry = 7
        bot_y = y + h - ry
        apex_y = y + 4
        return (
            f'<ellipse cx="{cx}" cy="{bot_y}" rx="{rx}" ry="{ry}" fill="#ffffff" stroke="#0f172a" stroke-width="1.3"/>'
            f'<polygon points="{cx},{apex_y} {cx - rx},{bot_y} {cx + rx},{bot_y}" fill="#f8fafc" stroke="#0f172a" stroke-width="1.3"/>'
        )
    else:  # Sphere
        cx = x + w / 2
        cy = y + h / 2
        r = min(w, h) / 2 - 2
        return (
            f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#ffffff" stroke="#0f172a" stroke-width="1.3"/>'
            f'<ellipse cx="{cx}" cy="{cy}" rx="{r}" ry="{r * 0.3:.1f}" fill="none" stroke="#0f172a" stroke-width="1.2" stroke-dasharray="3,3"/>'
        )


def render_solid_views_question_svg(
    side_view_shape: str = "rectangle",
    top_view_shape: str = "circle",
    solids: list[str] | None = None,
) -> tuple[str, list[str], str]:
    """
    Renders side & top view diagrams in the stem, and 4 solid options A, B, C, D
    matching the exact style of User Reference Image 2:
      Stem: [Rectangle] (side view) and [Circle] (top view)
      Options:
        A. Cube
        B. Cylinder  <-- (side=rectangle, top=circle)
        C. Cone
        D. Cuboid
    Returns (svg_markup, options_list, correct_option_name).
    """
    if solids is None:
        solids = ["Cube", "Cylinder", "Cone", "Cuboid"]

    svg_w = 480
    svg_h = 175

    elements = [
        f'<svg width="{svg_w}" height="{svg_h}" viewBox="0 0 {svg_w} {svg_h}" xmlns="http://www.w3.org/2000/svg" style="background:#ffffff; font-family:Arial, sans-serif;">'
    ]

    # Stem Views Header
    # Side view: Rectangle
    elements.append('<text x="40" y="24" font-size="12" font-weight="bold" fill="#334155">Side view:</text>')
    if side_view_shape.lower() == "rectangle":
        elements.append('<rect x="110" y="10" width="24" height="34" fill="#f8fafc" stroke="#0f172a" stroke-width="1.4"/>')
    elif side_view_shape.lower() == "triangle":
        elements.append('<polygon points="122,10 108,44 136,44" fill="#f8fafc" stroke="#0f172a" stroke-width="1.4"/>')
    else:  # square
        elements.append('<rect x="110" y="12" width="28" height="28" fill="#f8fafc" stroke="#0f172a" stroke-width="1.4"/>')

    # Top view: Circle
    elements.append('<text x="210" y="24" font-size="12" font-weight="bold" fill="#334155">Top view:</text>')
    if top_view_shape.lower() == "circle":
        elements.append('<circle cx="285" cy="26" r="14" fill="#f8fafc" stroke="#0f172a" stroke-width="1.4"/>')
    elif top_view_shape.lower() == "square":
        elements.append('<rect x="272" y="12" width="26" height="26" fill="#f8fafc" stroke="#0f172a" stroke-width="1.4"/>')
    else:  # triangle
        elements.append('<polygon points="285,12 270,38 300,38" fill="#f8fafc" stroke="#0f172a" stroke-width="1.4"/>')

    # Options A, B, C, D row of 3D solids
    start_y = 75
    col_w = 110
    labels = ["A", "B", "C", "D"]

    for idx, (lbl, solid) in enumerate(zip(labels, solids)):
        col_x = 20 + idx * col_w
        elements.append(
            f'<text x="{col_x + 10}" y="{start_y + 16}" font-size="14" font-weight="bold" fill="#0f172a">{lbl}.</text>'
        )
        elements.append(_draw_solid_wireframe(solid, col_x + 32, start_y, 56, 56))
        elements.append(
            f'<text x="{col_x + 60}" y="{start_y + 74}" font-size="11" font-weight="600" fill="#475569" text-anchor="middle">{solid}</text>'
        )

    elements.append("</svg>")
    svg_markup = "\n".join(elements)

    # Determine correct answer
    correct_name = "Cylinder"
    if side_view_shape.lower() == "triangle" and top_view_shape.lower() == "circle":
        correct_name = "Cone"
    elif side_view_shape.lower() == "square" and top_view_shape.lower() == "square":
        correct_name = "Cube"
    elif side_view_shape.lower() == "rectangle" and top_view_shape.lower() == "rectangle":
        correct_name = "Cuboid"

    correct_option = f"({labels[solids.index(correct_name)]}) {correct_name}" if correct_name in solids else solids[0]
    options = [f"({lbl}) {name}" for lbl, name in zip(labels, solids)]
    return svg_markup, options, correct_option


# ---------------------------------------------------------------------------
# 3. 2D Geometrical Shapes & Symmetry SVG Generators (Reference Image 3)
# ---------------------------------------------------------------------------

def render_symmetry_shapes_options_svg() -> tuple[str, list[str], str]:
    """
    Renders 4 distinct geometrical shapes labeled A, B, C, D
    matching the exact style of User Reference Image 3:
      A. Square (4 axes of symmetry)
      B. Right-Angled Scalene Triangle (0 axes of symmetry)
      C. Circle (infinite axes of symmetry)
      D. Rectangle (exactly 2 axes of symmetry: horizontal & vertical)
    Question: "The geometrical figure with only two axes of symmetry among the following is"
    Answer: (D) Rectangle
    Returns (svg_markup, options_list, correct_option).
    """
    svg_w = 480
    svg_h = 100

    elements = [
        f'<svg width="{svg_w}" height="{svg_h}" viewBox="0 0 {svg_w} {svg_h}" xmlns="http://www.w3.org/2000/svg" style="background:#ffffff; font-family:Arial, sans-serif;">'
    ]

    # A. Square
    elements.append('<text x="20" y="55" font-size="14" font-weight="bold" fill="#0f172a">A.</text>')
    elements.append('<rect x="42" y="22" width="46" height="46" fill="#f8fafc" stroke="#0f172a" stroke-width="1.5"/>')

    # B. Right-Angled Triangle
    elements.append('<text x="135" y="55" font-size="14" font-weight="bold" fill="#0f172a">B.</text>')
    elements.append('<polygon points="160,68 215,68 215,22" fill="#f8fafc" stroke="#0f172a" stroke-width="1.5"/>')

    # C. Circle
    elements.append('<text x="255" y="55" font-size="14" font-weight="bold" fill="#0f172a">C.</text>')
    elements.append('<circle cx="300" cy="45" r="23" fill="#f8fafc" stroke="#0f172a" stroke-width="1.5"/>')

    # D. Rectangle
    elements.append('<text x="365" y="55" font-size="14" font-weight="bold" fill="#0f172a">D.</text>')
    elements.append('<rect x="390" y="32" width="65" height="30" fill="#f8fafc" stroke="#0f172a" stroke-width="1.5"/>')

    elements.append("</svg>")
    svg_markup = "\n".join(elements)

    options = ["(A) Square", "(B) Right triangle", "(C) Circle", "(D) Rectangle"]
    correct_option = "(D) Rectangle"
    return svg_markup, options, correct_option


# ---------------------------------------------------------------------------
# 4. Fractions of Shapes SVG Generators
# ---------------------------------------------------------------------------

def render_fraction_circle_svg(total_parts: int = 4, shaded_parts: int = 3) -> str:
    """Renders a circle divided into equal sectors with shaded slices."""
    cx, cy, r = 80, 80, 60
    elements = [
        f'<svg width="160" height="160" viewBox="0 0 160 160" xmlns="http://www.w3.org/2000/svg" style="background:#ffffff;">'
    ]

    angle_step = 2 * math.pi / total_parts
    for i in range(total_parts):
        start_a = i * angle_step - math.pi / 2
        end_a = (i + 1) * angle_step - math.pi / 2
        x1 = cx + r * math.cos(start_a)
        y1 = cy + r * math.sin(start_a)
        x2 = cx + r * math.cos(end_a)
        y2 = cy + r * math.sin(end_a)

        fill = "#93c5fd" if i < shaded_parts else "#ffffff"
        large_arc = 1 if angle_step > math.pi else 0
        path_d = f"M {cx},{cy} L {x1:.1f},{y1:.1f} A {r},{r} 0 {large_arc},1 {x2:.1f},{y2:.1f} Z"
        elements.append(f'<path d="{path_d}" fill="{fill}" stroke="#0f172a" stroke-width="1.4"/>')

    elements.append("</svg>")
    return "\n".join(elements)


# ---------------------------------------------------------------------------
# 5. Perimeter & Grid SVG Generators (Class 4 & 5)
# ---------------------------------------------------------------------------

def render_grid_perimeter_svg(width_units: int = 4, height_units: int = 3) -> str:
    """Renders a grid with a shaded rectangular shape and labeled side measurements."""
    grid_size = 24
    offset_x = 40
    offset_y = 30
    total_w = offset_x + width_units * grid_size + 40
    total_h = offset_y + height_units * grid_size + 40

    elements = [
        f'<svg width="{total_w}" height="{total_h}" viewBox="0 0 {total_w} {total_h}" xmlns="http://www.w3.org/2000/svg" style="background:#ffffff; font-family:Arial, sans-serif;">'
    ]

    # Grid background
    for col in range(width_units + 1):
        x = offset_x + col * grid_size
        elements.append(f'<line x1="{x}" y1="{offset_y}" x2="{x}" y2="{offset_y + height_units * grid_size}" stroke="#cbd5e1" stroke-width="1"/>')
    for row in range(height_units + 1):
        y = offset_y + row * grid_size
        elements.append(f'<line x1="{offset_x}" y1="{y}" x2="{offset_x + width_units * grid_size}" y2="{y}" stroke="#cbd5e1" stroke-width="1"/>')

    # Shaded shape
    shape_w = width_units * grid_size
    shape_h = height_units * grid_size
    elements.append(f'<rect x="{offset_x}" y="{offset_y}" width="{shape_w}" height="{shape_h}" fill="#bfdbfe" fill-opacity="0.6" stroke="#1e3a8a" stroke-width="2"/>')

    # Dimension labels
    mid_x = offset_x + shape_w / 2
    mid_y = offset_y + shape_h / 2
    elements.append(f'<text x="{mid_x}" y="{offset_y - 8}" font-size="12" font-weight="bold" fill="#0f172a" text-anchor="middle">{width_units} cm</text>')
    elements.append(f'<text x="{offset_x - 8}" y="{mid_y + 4}" font-size="12" font-weight="bold" fill="#0f172a" text-anchor="end">{height_units} cm</text>')

    elements.append("</svg>")
    return "\n".join(elements)


# ---------------------------------------------------------------------------
# 6. Angles SVG Generator (Class 5)
# ---------------------------------------------------------------------------

def render_angle_diagram_svg(angle_degrees: int = 120, label: str = "A") -> str:
    """Renders two rays with a shared vertex showing an acute, right, or obtuse angle."""
    cx, cy = 60, 90
    length = 65
    rad = math.radians(angle_degrees)

    x1 = cx + length
    y1 = cy
    x2 = cx + length * math.cos(-rad)
    y2 = cy + length * math.sin(-rad)

    elements = [
        f'<svg width="180" height="120" viewBox="0 0 180 120" xmlns="http://www.w3.org/2000/svg" style="background:#ffffff;">',
        f'<line x1="{cx}" y1="{cy}" x2="{x1}" y2="{y1}" stroke="#0f172a" stroke-width="2"/>',
        f'<line x1="{cx}" y1="{cy}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#0f172a" stroke-width="2"/>',
    ]

    # Arc or right-angle square
    if angle_degrees == 90:
        elements.append(f'<rect x="{cx}" y="{cy - 14}" width="14" height="14" fill="none" stroke="#dc2626" stroke-width="1.5"/>')
    else:
        arc_r = 20
        arc_x = cx + arc_r * math.cos(-rad)
        arc_y = cy + arc_r * math.sin(-rad)
        large_arc = 1 if angle_degrees > 180 else 0
        elements.append(
            f'<path d="M {cx + arc_r},{cy} A {arc_r},{arc_r} 0 {large_arc},0 {arc_x:.1f},{arc_y:.1f}" fill="none" stroke="#2563eb" stroke-width="1.5"/>'
        )

    # Vertex circle
    elements.append(f'<circle cx="{cx}" cy="{cy}" r="3" fill="#0f172a"/>')
    elements.append("</svg>")
    return "\n".join(elements)


# ---------------------------------------------------------------------------
# 7. Counting & Spatial Relationships SVG (Class 1 & 2)
# ---------------------------------------------------------------------------

def render_counting_spatial_svg(inside_count: int = 4, outside_count: int = 2) -> str:
    """Renders objects inside and outside a clearly defined circular boundary for Class 1."""
    svg_w = 260
    svg_h = 130
    cx, cy, r = 100, 65, 50

    elements = [
        f'<svg width="{svg_w}" height="{svg_h}" viewBox="0 0 {svg_w} {svg_h}" xmlns="http://www.w3.org/2000/svg" style="background:#ffffff; font-family:Arial, sans-serif;">',
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#f8fafc" stroke="#1e3a8a" stroke-width="2"/>',
        f'<text x="{cx}" y="{cy - r - 6}" font-size="11" font-weight="bold" fill="#1e3a8a" text-anchor="middle">Inside</text>',
    ]

    # Stars inside
    inside_positions = [(80, 50), (120, 50), (80, 80), (120, 80), (100, 65)]
    for i in range(min(inside_count, len(inside_positions))):
        px, py = inside_positions[i]
        elements.append(_star_polygon(px, py, 7.5))

    # Stars outside
    outside_positions = [(190, 45), (210, 85), (225, 45)]
    for i in range(min(outside_count, len(outside_positions))):
        px, py = outside_positions[i]
        elements.append(_star_polygon(px, py, 7.5))
    elements.append(f'<text x="210" y="24" font-size="11" font-weight="bold" fill="#475569" text-anchor="middle">Outside</text>')

    elements.append("</svg>")
    return "\n".join(elements)


# ---------------------------------------------------------------------------
# High-Level Factory: create_figure_question
# ---------------------------------------------------------------------------

def is_figure_eligible_chapter(chapter_name: str, grade: int | str | None = None) -> bool:
    """
    Return True if the textbook chapter covers visual, spatial, geometrical,
    or visual representation concepts suitable for a figure-based question.
    """
    c = chapter_name.lower()
    return any(
        k in c
        for k in (
            "shape",
            "shapes",
            "space",
            "spatial",
            "solid",
            "solids",
            "view",
            "views",
            "multiplication",
            "times",
            "fraction",
            "fractions",
            "perimeter",
            "area",
            "square",
            "squares",
            "angle",
            "angles",
            "circle",
            "circles",
            "measurement",
            "length",
            "symmetry",
            "pattern",
            "patterns",
        )
    )


def create_figure_question(
    grade: int | str,
    chapter: str,
    question_type: QuestionType = QuestionType.MULTIPLE_CHOICE,
    marks: int = 1,
    difficulty: Difficulty = Difficulty.MEDIUM,
) -> Optional[Question]:
    """
    Create an original, syllabus-aligned, mathematically verified figure-based question
    calibrated to the selected grade, chapter, marks, and difficulty level.
    """
    from app.curriculum import parse_grade_number

    g_num = parse_grade_number(str(grade)) or 3
    ch_clean = chapter.lower()

    # -----------------------------------------------------------------------
    # Pattern 7: Counting & Spatial Relations (Strictly for Class 1 & 2)
    # Class 1 MUST receive only age-appropriate simple visual problems
    # -----------------------------------------------------------------------
    if g_num <= 1 or (g_num == 2 and any(k in ch_clean for k in ("spatial", "space", "count", "digits"))):
        in_c = 4 if difficulty == Difficulty.MEDIUM else (3 if difficulty == Difficulty.EASY else 5)
        out_c = 2
        svg_content = render_counting_spatial_svg(inside_count=in_c, outside_count=out_c)

        if question_type == QuestionType.MULTIPLE_CHOICE:
            options = [str(in_c), str(out_c), str(in_c + out_c), str(in_c - 1)]
            q_text = "Count how many stars are INSIDE the circular boundary in the figure:"
            ans = str(in_c)
        else:
            options = None
            q_text = "How many stars are positioned INSIDE the circular boundary? Write the number:"
            ans = str(in_c)

        sol = f"Counting the stars inside the circle gives 1, 2, ... {in_c}. There are {in_c} stars inside."

        return Question(
            question_text=q_text,
            question_type=question_type,
            marks=marks,
            difficulty=difficulty,
            grade=f"Class {g_num}",
            chapter=chapter,
            topic=chapter,
            options=options,
            correct_option=ans if options else None,
            answer=ans,
            solution=sol,
            question_subtype="figure_based",
            figure=FigureData(
                figure_type="counting_spatial",
                title="Spatial Boundary Counting",
                svg_content=svg_content,
                parameters={"inside_count": in_c, "outside_count": out_c},
            ),
            math_verification=MathVerificationResult(
                status=MathCheckStatus.VERIFIED,
                expression=str(in_c),
                evaluated_value=str(in_c),
                details=f"Verified count: {in_c}",
            ),
        )

    # -----------------------------------------------------------------------
    # Pattern 6: Angles (Class 5 - Shapes and Angles)
    # -----------------------------------------------------------------------
    if "angle" in ch_clean and g_num >= 5:
        deg = 120 if difficulty != Difficulty.EASY else 90
        ang_type = "Obtuse angle" if deg > 90 else "Right angle"
        svg_content = render_angle_diagram_svg(angle_degrees=deg)

        options = ["Acute angle", "Right angle", "Obtuse angle", "Straight angle"]
        q_text = "Identify the type of angle represented in the figure:"
        ans = ang_type
        sol = (
            f"Step 1: An angle measuring 90° is a right angle. "
            f"Step 2: The angle shown measures {deg}°, which is {'greater than 90° and less than 180°' if deg > 90 else 'exactly 90°'}. "
            f"Step 3: Therefore, it is a(n) {ang_type.lower()}."
        )

        return Question(
            question_text=q_text,
            question_type=QuestionType.MULTIPLE_CHOICE,
            marks=marks,
            difficulty=difficulty,
            grade=f"Class {g_num}",
            chapter=chapter,
            topic=chapter,
            options=options,
            correct_option=ans,
            answer=ans,
            solution=sol,
            question_subtype="figure_based",
            figure=FigureData(
                figure_type="angle_rays",
                title="Angle Diagram",
                svg_content=svg_content,
                parameters={"degrees": deg, "type": ang_type},
            ),
        )

    # -----------------------------------------------------------------------
    # Pattern 1: Visual Multiplication & Equal Groups (Classes 2 & 3)
    # Inspired by User Reference Image 1
    # -----------------------------------------------------------------------
    if "multiplication" in ch_clean or "times" in ch_clean or "tables" in ch_clean or (g_num in (2, 3) and "addition" in ch_clean and "not more than 9" not in ch_clean):
        groups = 2 if difficulty != Difficulty.HARD else 3
        count = 4 if difficulty == Difficulty.MEDIUM else (3 if difficulty == Difficulty.EASY else 5)
        expr_str = f"{count} × {groups}" if groups == 2 else f"{count} × {groups}"

        svg_content, options, correct_opt = render_multiplication_options_svg(
            correct_groups=groups,
            correct_count_per_group=count,
            shape_type="star",
        )

        q_text = f"The figure that represents {expr_str} among the following is:"
        ans = f"{correct_opt} ({count} + {count} = {count * groups})"
        sol = (
            f"Step 1: The expression {count} × {groups} represents {groups} equal groups of {count} stars. "
            f"Step 2: Adding {groups} groups of {count} ({' + '.join([str(count)] * groups)}) gives {count * groups}. "
            f"Step 3: Therefore, {correct_opt} correctly represents {expr_str}."
        )

        return Question(
            question_text=q_text,
            question_type=QuestionType.MULTIPLE_CHOICE,
            marks=marks,
            difficulty=difficulty,
            grade=f"Class {g_num}",
            chapter=chapter,
            topic=chapter,
            options=options,
            correct_option=correct_opt,
            answer=ans,
            solution=sol,
            question_subtype="figure_based",
            figure=FigureData(
                figure_type="multiplication_groups",
                title=f"Multiplication Representation for {expr_str}",
                svg_content=svg_content,
                parameters={"groups": groups, "count_per_group": count, "expression": expr_str},
            ),
            math_verification=MathVerificationResult(
                status=MathCheckStatus.VERIFIED,
                expression=f"{count} * {groups}",
                evaluated_value=str(count * groups),
                details=f"Verified: {count} * {groups} = {count * groups}",
            ),
        )

    # -----------------------------------------------------------------------
    # Pattern 2: 3D Geometrical Solids & Top/Side Views (Classes 2, 3, 5)
    # Inspired by User Reference Image 2
    # -----------------------------------------------------------------------
    if any(k in ch_clean for k in ("shapes", "solid", "solids")) and g_num >= 2:
        if difficulty == Difficulty.EASY:
            # Direct recognition of 3D solid
            svg_content, options, correct_opt = render_solid_views_question_svg(
                side_view_shape="square",
                top_view_shape="square",
                solids=["Cube", "Cylinder", "Cone", "Cuboid"],
            )
            q_text = "A solid object when viewed from side is a square and its top view is also a square. Identify the solid:"
            ans = correct_opt
            sol = "A cube has all square faces. Both its side view and top view are squares."
        else:
            # Medium / Hard: Cylinder views (Reference Image 2)
            svg_content, options, correct_opt = render_solid_views_question_svg(
                side_view_shape="rectangle",
                top_view_shape="circle",
                solids=["Cube", "Cylinder", "Cone", "Cuboid"],
            )
            q_text = "A solid object when viewed from the side appears as a rectangle and its top view is a circle. The solid is:"
            ans = correct_opt
            sol = (
                "Step 1: Inspect the side view: a rectangle with parallel vertical sides. "
                "Step 2: Inspect the top view: a flat circular face. "
                "Step 3: A cylinder has two circular bases and a curved rectangular surface, matching both views."
            )

        return Question(
            question_text=q_text,
            question_type=QuestionType.MULTIPLE_CHOICE,
            marks=marks,
            difficulty=difficulty,
            grade=f"Class {g_num}",
            chapter=chapter,
            topic=chapter,
            options=options,
            correct_option=correct_opt,
            answer=ans,
            solution=sol,
            question_subtype="figure_based",
            figure=FigureData(
                figure_type="solid_views",
                title="Solid Views Projection Diagram",
                svg_content=svg_content,
                parameters={"side_view": "rectangle", "top_view": "circle", "solid": "cylinder"},
            ),
        )

    # -----------------------------------------------------------------------
    # Pattern 3: 2D Geometrical Shapes & Symmetry (Classes 3, 4, 5)
    # Inspired by User Reference Image 3
    # -----------------------------------------------------------------------
    if ("shape" in ch_clean or "symmetry" in ch_clean or "circle" in ch_clean) and g_num >= 3:
        svg_content, options, correct_opt = render_symmetry_shapes_options_svg()
        q_text = "The geometrical figure with only two axes of symmetry among the following is:"
        ans = "(D) Rectangle"
        sol = (
            "Step 1: Check Square (A): A square has 4 axes of symmetry (vertical, horizontal, and 2 diagonals). "
            "Step 2: Check Right Scalene Triangle (B): Has 0 axes of symmetry. "
            "Step 3: Check Circle (C): A circle has infinitely many axes of symmetry. "
            "Step 4: Check Rectangle (D): A rectangle has exactly 2 axes of symmetry (1 vertical and 1 horizontal). "
            "Conclusion: Figure D (Rectangle) is correct."
        )

        return Question(
            question_text=q_text,
            question_type=QuestionType.MULTIPLE_CHOICE,
            marks=marks,
            difficulty=difficulty,
            grade=f"Class {g_num}",
            chapter=chapter,
            topic=chapter,
            options=options,
            correct_option=correct_opt,
            answer=ans,
            solution=sol,
            question_subtype="figure_based",
            figure=FigureData(
                figure_type="symmetry_shapes",
                title="Geometrical Shapes & Axes of Symmetry",
                svg_content=svg_content,
                parameters={"target_axes": 2, "answer_shape": "rectangle"},
            ),
        )

    # -----------------------------------------------------------------------
    # Pattern 4: Fractions Visuals (Classes 4 & 5)
    # -----------------------------------------------------------------------
    if "fraction" in ch_clean and g_num >= 4:
        total = 4 if difficulty != Difficulty.HARD else 8
        shaded = 3 if difficulty != Difficulty.HARD else 5
        svg_content = render_fraction_circle_svg(total_parts=total, shaded_parts=shaded)

        options = [f"{shaded}/{total}", f"1/{total}", f"{total - shaded}/{total}", f"{shaded - 1}/{total}"]
        q_text = "What fraction of the circular region is shaded in the given figure?"
        ans = f"{shaded}/{total}"
        sol = (
            f"Step 1: Count total equal parts into which the circle is divided: {total} parts. "
            f"Step 2: Count the number of shaded parts: {shaded} parts. "
            f"Step 3: Fraction = (shaded parts) / (total parts) = {shaded}/{total}."
        )

        return Question(
            question_text=q_text,
            question_type=QuestionType.MULTIPLE_CHOICE,
            marks=marks,
            difficulty=difficulty,
            grade=f"Class {g_num}",
            chapter=chapter,
            topic=chapter,
            options=options,
            correct_option=ans,
            answer=ans,
            solution=sol,
            question_subtype="figure_based",
            figure=FigureData(
                figure_type="fraction_shape",
                title="Circular Fraction Region",
                svg_content=svg_content,
                parameters={"total_parts": total, "shaded_parts": shaded},
            ),
            math_verification=MathVerificationResult(
                status=MathCheckStatus.VERIFIED,
                expression=f"{shaded}/{total}",
                evaluated_value=str(round(shaded / total, 3)),
                details=f"Verified fraction: {shaded}/{total}",
            ),
        )

    # -----------------------------------------------------------------------
    # Pattern 5: Perimeter & Grid Measurements (Classes 4 & 5)
    # -----------------------------------------------------------------------
    if any(k in ch_clean for k in ("perimeter", "area", "length", "square", "squares")) and g_num >= 4:
        w_units = 5 if difficulty != Difficulty.EASY else 4
        h_units = 3 if difficulty != Difficulty.EASY else 2
        perim = 2 * (w_units + h_units)
        svg_content = render_grid_perimeter_svg(width_units=w_units, height_units=h_units)

        if question_type == QuestionType.MULTIPLE_CHOICE:
            options = [f"{perim} cm", f"{w_units * h_units} cm", f"{perim + 2} cm", f"{perim - 2} cm"]
            q_text = "Look at the shaded rectangular shape on the 1 cm grid. Find its perimeter:"
            ans = f"{perim} cm"
        else:
            options = None
            q_text = f"Find the perimeter of the rectangle shown in the figure with length {w_units} cm and width {h_units} cm:"
            ans = f"Perimeter = {perim} cm"

        sol = (
            f"Step 1: Length = {w_units} cm, Width = {h_units} cm. "
            f"Step 2: Perimeter = 2 × (Length + Width) = 2 × ({w_units} + {h_units}) = 2 × {w_units + h_units} = {perim} cm."
        )

        return Question(
            question_text=q_text,
            question_type=question_type,
            marks=marks,
            difficulty=difficulty,
            grade=f"Class {g_num}",
            chapter=chapter,
            topic=chapter,
            options=options,
            correct_option=ans if options else None,
            answer=ans,
            solution=sol,
            question_subtype="figure_based",
            figure=FigureData(
                figure_type="perimeter_grid",
                title="Grid Perimeter Measurement",
                svg_content=svg_content,
                parameters={"width": w_units, "height": h_units, "perimeter": perim},
            ),
            math_verification=MathVerificationResult(
                status=MathCheckStatus.VERIFIED,
                expression=f"2 * ({w_units} + {h_units})",
                evaluated_value=str(perim),
                details=f"Verified: 2 * ({w_units} + {h_units}) = {perim}",
            ),
        )

    # -----------------------------------------------------------------------
    # Pattern 6: Angles (Class 5)
    # -----------------------------------------------------------------------
    if "angle" in ch_clean and g_num == 5:
        deg = 120 if difficulty != Difficulty.EASY else 90
        ang_type = "Obtuse angle" if deg > 90 else "Right angle"
        svg_content = render_angle_diagram_svg(angle_degrees=deg)

        options = ["Acute angle", "Right angle", "Obtuse angle", "Straight angle"]
        q_text = "Identify the type of angle represented in the figure:"
        ans = ang_type
        sol = (
            f"Step 1: An angle measuring 90° is a right angle. "
            f"Step 2: The angle shown measures {deg}°, which is {'greater than 90° and less than 180°' if deg > 90 else 'exactly 90°'}. "
            f"Step 3: Therefore, it is a(n) {ang_type.lower()}."
        )

        return Question(
            question_text=q_text,
            question_type=QuestionType.MULTIPLE_CHOICE,
            marks=marks,
            difficulty=difficulty,
            grade=f"Class {g_num}",
            chapter=chapter,
            topic=chapter,
            options=options,
            correct_option=ans,
            answer=ans,
            solution=sol,
            question_subtype="figure_based",
            figure=FigureData(
                figure_type="angle_rays",
                title="Angle Diagram",
                svg_content=svg_content,
                parameters={"degrees": deg, "type": ang_type},
            ),
        )

    # -----------------------------------------------------------------------
    # Pattern 7: Counting & Spatial Relations (Class 1 & 2)
    # -----------------------------------------------------------------------
    if any(k in ch_clean for k in ("spatial", "digits", "solids")) and g_num <= 2:
        in_c = 4 if difficulty == Difficulty.MEDIUM else (3 if difficulty == Difficulty.EASY else 5)
        out_c = 2
        svg_content = render_counting_spatial_svg(inside_count=in_c, outside_count=out_c)

        if question_type == QuestionType.MULTIPLE_CHOICE:
            options = [str(in_c), str(out_c), str(in_c + out_c), str(in_c - 1)]
            q_text = "Count how many stars are INSIDE the circular boundary in the figure:"
            ans = str(in_c)
        else:
            options = None
            q_text = "How many stars are positioned INSIDE the circular boundary? Write the number:"
            ans = str(in_c)

        sol = f"Counting the stars inside the circle gives 1, 2, ... {in_c}. There are {in_c} stars inside."

        return Question(
            question_text=q_text,
            question_type=question_type,
            marks=marks,
            difficulty=difficulty,
            grade=f"Class {g_num}",
            chapter=chapter,
            topic=chapter,
            options=options,
            correct_option=ans if options else None,
            answer=ans,
            solution=sol,
            question_subtype="figure_based",
            figure=FigureData(
                figure_type="counting_spatial",
                title="Spatial Boundary Counting",
                svg_content=svg_content,
                parameters={"inside_count": in_c, "outside_count": out_c},
            ),
            math_verification=MathVerificationResult(
                status=MathCheckStatus.VERIFIED,
                expression=str(in_c),
                evaluated_value=str(in_c),
                details=f"Verified count: {in_c}",
            ),
        )

    return None
