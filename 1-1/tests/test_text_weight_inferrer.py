from PIL import Image, ImageDraw

from docflow.appearance.text_weight_inferrer import infer_text_stroke_ratio


def _stroke_sample(width: int) -> Image.Image:
    image = Image.new("RGB", (120, 40), "white")
    draw = ImageDraw.Draw(image)
    for left in (15, 45, 75, 105):
        draw.line((left, 5, left, 34), fill="black", width=width)
        draw.line((left - 8, 12, left + 8, 12), fill="black", width=width)
        draw.line((left - 8, 27, left + 8, 27), fill="black", width=width)
    return image


def test_stroke_ratio_distinguishes_bold_from_regular_at_the_same_scale() -> None:
    regular = infer_text_stroke_ratio((_stroke_sample(2),))
    bold = infer_text_stroke_ratio((_stroke_sample(5),))

    assert regular is not None
    assert bold is not None
    assert bold > regular * 1.5
