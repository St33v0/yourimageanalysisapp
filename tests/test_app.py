from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import app
from azure.ai.vision.imageanalysis.models import VisualFeatures


DEFAULT_IMAGE_URL = (
    "https://raw.githubusercontent.com/Azure-Samples/"
    "cognitive-services-sample-data-files/master/ComputerVision/Images/faces.jpg"
)


@pytest.fixture
def streamlit_mocks(monkeypatch):
    mocks = SimpleNamespace(
        markdown=Mock(),
        text=Mock(),
        image=Mock(),
        warning=Mock(),
        error=Mock(),
        selectbox=Mock(),
        text_input=Mock(),
        file_uploader=Mock(),
        button=Mock(),
    )
    for name in (
        "markdown",
        "text",
        "image",
        "warning",
        "error",
        "selectbox",
        "text_input",
        "file_uploader",
        "button",
    ):
        monkeypatch.setattr(app.st, name, getattr(mocks, name))
    return mocks


@pytest.fixture
def vision_client(monkeypatch):
    result = SimpleNamespace(
        caption=SimpleNamespace(text="A test image", confidence=0.9),
        tags=SimpleNamespace(
            list=[SimpleNamespace(name="test", confidence=0.8)]
        ),
    )
    client = Mock()
    client.analyze.return_value = result
    client.analyze_from_url.return_value = result
    monkeypatch.setattr(app, "client", client)
    return client


def test_analyzes_image_url(streamlit_mocks, vision_client):
    image_url = "https://example.com/image.jpg"
    streamlit_mocks.selectbox.return_value = "URL"
    streamlit_mocks.text_input.return_value = image_url
    streamlit_mocks.button.return_value = True

    app.main()

    vision_client.analyze_from_url.assert_called_once_with(
        image_url=image_url,
        visual_features=[VisualFeatures.TAGS, VisualFeatures.CAPTION],
    )
    vision_client.analyze.assert_not_called()
    streamlit_mocks.image.assert_called_once_with(
        image_url, caption="Image to analyse"
    )


def test_analyzes_local_file(streamlit_mocks, vision_client, tmp_path):
    image_data = b"local image"
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(image_data)
    streamlit_mocks.selectbox.return_value = "Local file path"
    streamlit_mocks.text_input.return_value = str(image_path)
    streamlit_mocks.button.return_value = True

    app.main()

    vision_client.analyze.assert_called_once_with(
        image_data=image_data,
        visual_features=[VisualFeatures.TAGS, VisualFeatures.CAPTION],
    )
    vision_client.analyze_from_url.assert_not_called()
    streamlit_mocks.image.assert_called_once_with(
        image_data, caption="Image to analyse"
    )


def test_analyzes_uploaded_image(streamlit_mocks, vision_client):
    image_data = b"uploaded image"
    uploaded_image = Mock()
    uploaded_image.getvalue.return_value = image_data
    streamlit_mocks.selectbox.return_value = "Upload image"
    streamlit_mocks.file_uploader.return_value = uploaded_image
    streamlit_mocks.button.return_value = True

    app.main()

    vision_client.analyze.assert_called_once_with(
        image_data=image_data,
        visual_features=[VisualFeatures.TAGS, VisualFeatures.CAPTION],
    )
    vision_client.analyze_from_url.assert_not_called()
    streamlit_mocks.file_uploader.assert_called_once_with(
        "Choose an image",
        type=["jpg", "jpeg", "png", "gif", "bmp"],
    )
    uploaded_image.getvalue.assert_called_once_with()


@pytest.mark.parametrize(
    ("source", "text_input", "uploaded_image"),
    [
        ("URL", "", None),
        ("Local file path", "/path/that/does/not/exist.jpg", None),
        ("Upload image", "", None),
    ],
)
def test_reports_missing_or_invalid_image(
    streamlit_mocks,
    vision_client,
    source,
    text_input,
    uploaded_image,
):
    streamlit_mocks.selectbox.return_value = source
    streamlit_mocks.text_input.return_value = text_input
    streamlit_mocks.file_uploader.return_value = uploaded_image
    streamlit_mocks.button.return_value = True

    app.main()

    vision_client.analyze.assert_not_called()
    vision_client.analyze_from_url.assert_not_called()
    streamlit_mocks.error.assert_called_once()
    if source == "Local file path":
        streamlit_mocks.warning.assert_called_once_with(
            "Enter a path to an image file available to the app."
        )
