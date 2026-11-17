from pathlib import Path

import streamlit as st
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

import os
from dotenv import load_dotenv

# loading variables from .env file
load_dotenv()

# accessing vision key and endpoint
vision_key = os.getenv("VISION_KEY")
vision_endpoint = os.getenv("VISION_ENDPOINT")

# Create the Image Analysis client
client = ImageAnalysisClient(
    endpoint=vision_endpoint,
    credential=AzureKeyCredential(vision_key)
)

def main():
    st.markdown("# Image Analysis App 🖼️")
    st.markdown("### This app allows you to extract visual features from images.")

    image_source = st.selectbox(
        "Image source",
        ["URL", "Local file path", "Upload image"],
    )

    image_url = None
    image_data = None
    image_caption = "Image to analyse"

    if image_source == "URL":
        image_url = st.text_input(
            "Enter a direct image URL",
            "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-sample-data-files/master/ComputerVision/Images/faces.jpg",
        )
        if image_url:
            st.image(image_url, caption=image_caption)
    elif image_source == "Local file path":
        image_path = st.text_input("Enter a local image file path")
        if image_path:
            path = Path(image_path).expanduser()
            if path.is_file():
                image_data = path.read_bytes()
                st.image(image_data, caption=image_caption)
            else:
                st.warning("Enter a path to an image file available to the app.")
    else:
        uploaded_image = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png", "gif", "bmp"],
        )
        if uploaded_image is not None:
            image_data = uploaded_image.getvalue()
            st.image(image_data, caption=image_caption)

    if st.button("Analyse image"):
        if image_source == "URL":
            if not image_url:
                st.error("Enter an image URL first.")
                return
            result = client.analyze_from_url(
                image_url=image_url,
                visual_features=[VisualFeatures.TAGS, VisualFeatures.CAPTION],
            )
        else:
            if not image_data:
                st.error("Choose or enter an image first.")
                return
            result = client.analyze(
                image_data=image_data,
                visual_features=[VisualFeatures.TAGS, VisualFeatures.CAPTION],
            )

        # Caption: a one-sentence description of the image
        if result.caption is not None:
            st.markdown(f"**Caption:** {result.caption.text} "
                        f"(confidence {result.caption.confidence:.2%})")

        # Tags: detected concepts with confidence
        st.markdown("**Tags detected:**")
        if result.tags is not None:
            for tag in result.tags.list:
                st.text(f"'{tag.name}' — confidence {tag.confidence:.2%}")

if __name__=='__main__':
    main()