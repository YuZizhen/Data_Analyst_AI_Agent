"""Render calculated results as self-contained PNG reports."""

from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock
from typing import Callable
from uuid import uuid4

from matplotlib import font_manager, rc_context
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import ScalarFormatter

_render_lock = Lock()
_available_fonts = {font.name for font in font_manager.fontManager.ttflist}
_font_families = [
    font
    for font in ("DejaVu Sans", "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC")
    if font in _available_fonts
]


def render_report(
    output_dir: Path,
    kind: str,
    title: str,
    subtitle: str,
    source: str,
    draw: Callable[[Axes], None],
    *,
    height: float = 5.6,
) -> dict[str, str]:
    """Write a complete, uniquely named image and return its path and title."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{kind}-{uuid4().hex[:12]}.png"

    with _render_lock, rc_context({"font.family": _font_families, "text.parse_math": False}):
        figure = Figure(figsize=(10.5, height), layout="constrained", facecolor="#f4f7fb")
        FigureCanvasAgg(figure)
        header, axes, footer = figure.subplots(3, 1, height_ratios=(0.95, height - 1.5, 0.45))
        header.set_axis_off()
        header.text(0.01, 0.65, title, fontsize=19, fontweight="bold", color="#102a43")
        header.text(0.01, 0.22, subtitle, fontsize=10, color="#486581")
        footer.set_axis_off()
        footer.text(0.01, 0.15, f"Source: {source}", fontsize=9, color="#486581")

        axes.set_facecolor("white")
        axes.set_axisbelow(True)
        axes.grid(axis="both", color="#e4ecf4", linewidth=0.8)
        axes.spines[:].set_visible(False)
        axes.tick_params(color="#9fb3c8", labelcolor="#243b53")
        for axis in (axes.xaxis, axes.yaxis):
            axis.set_major_formatter(ScalarFormatter(useOffset=False))
            axis.label.set_color("#243b53")
        draw(axes)

        with NamedTemporaryFile(dir=output_dir, suffix=".tmp", delete=False) as temporary:
            temporary_path = Path(temporary.name)
        try:
            figure.savefig(temporary_path, format="png", dpi=155, facecolor=figure.get_facecolor())
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)

    return {"title": title, "path": str(path.resolve())}
