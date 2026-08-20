import base64
import io

from PIL import Image

from docflow.analysis import DocumentAnalyzer
from docflow.appearance.font_classifier import FONT_FAMILY_BY_LABEL, FontClassifier, FontPrediction
from docflow.model.stages import RecognitionEvidence, RecognitionItem, RecognitionPage, Rect, TextEvidence


def _chinese_evidence() -> RecognitionEvidence:
    image = Image.new("RGB", (120, 36), "white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    item = RecognitionItem(
        "body",
        "text",
        Rect(10, 10, 130, 46),
        0,
        text_lines=(TextEvidence("正文文本"),),
        image_base64=base64.b64encode(buffer.getvalue()).decode("ascii"),
    )
    return RecognitionEvidence((RecognitionPage(0, 200, 100, (item,)),))


def test_analyzer_applies_only_an_accepted_supported_font() -> None:
    class Classifier:
        def predict_image(self, image):
            return FontPrediction("楷体", 0.91, 0.42, {"楷体": 0.91}, True)

    analysis = DocumentAnalyzer(Classifier()).analyze(_chinese_evidence())

    assert analysis.roles[0].font_family == "楷体"
    assert set(filter(None, FONT_FAMILY_BY_LABEL.values())) == {"宋体", "黑体", "楷体", "仿宋"}


def test_analyzer_disables_a_failed_optional_classifier() -> None:
    class Classifier:
        def predict_image(self, image):
            raise RuntimeError("model unavailable")

    analyzer = DocumentAnalyzer(Classifier())
    analysis = analyzer.analyze(_chinese_evidence())

    assert analysis.roles[0].font_family == "宋体"
    assert analyzer.font_classifier is None


def test_font_classifier_defaults_are_the_trained_acceptance_contract() -> None:
    classifier = FontClassifier("")

    assert classifier.reject_threshold == 0.6
    assert classifier.margin_threshold == 0.25
    assert classifier.transform.grayscale is True
    assert classifier.checkpoint_path.name == "mobilenetv3.xml"
