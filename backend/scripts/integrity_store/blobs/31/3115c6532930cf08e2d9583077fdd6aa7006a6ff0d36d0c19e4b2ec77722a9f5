from pydantic import BaseModel


class Rect(BaseModel):
    """Axis-aligned pixel rectangle, top-left origin."""

    x: float
    y: float
    w: float
    h: float


def quad_to_rect(quad: list[list[float]]) -> Rect:
    """deepreef-ocr returns 4-point quads `[[x,y],...]`; the review UI just
    needs an axis-aligned box to draw, so collapse to the bounding rect."""
    xs = [pt[0] for pt in quad]
    ys = [pt[1] for pt in quad]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    return Rect(x=x0, y=y0, w=x1 - x0, h=y1 - y0)
