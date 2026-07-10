from analyzer.parsing.ocr_parser import _orient_image


class FakeImage:
    def __init__(self):
        self.rotations = []

    def rotate(self, angle, *, expand):
        self.rotations.append((angle, expand))
        return "rotated"


class FakeTesseract:
    def __init__(self, osd=None, error=None):
        self.osd = osd
        self.error = error

    def image_to_osd(self, image):
        if self.error:
            raise self.error
        return self.osd


def test_orient_image_applies_inverse_of_osd_clockwise_correction():
    image = FakeImage()
    result = _orient_image(
        image,
        FakeTesseract("Orientation in degrees: 90\nRotate: 270\n"),
    )
    assert result == "rotated"
    assert image.rotations == [(90, True)]


def test_orient_image_leaves_upright_image_unchanged():
    image = FakeImage()
    assert _orient_image(image, FakeTesseract("Rotate: 0\n")) is image
    assert image.rotations == []


def test_orient_image_tolerates_osd_failure():
    image = FakeImage()
    assert _orient_image(image, FakeTesseract(error=RuntimeError("no text"))) is image
