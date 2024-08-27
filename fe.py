import streamlit as st
import requests
from PIL import Image, ImageDraw, ImageFont
import io
import pypdfium2 as pdfium
import numpy as np


def convertPdfToImage(pdf_bytes):
    pdf = pdfium.PdfDocument(pdf_bytes)
    n_pages = len(pdf)
    for page_number in range(n_pages):
        page = pdf.get_page(page_number)
        pil_image = page.render(
            scale=1,
            rotation=0,
            crop=(0, 0, 0, 0),
            grayscale=True
        )
    image = pil_image.to_pil()
    return image

# Function to draw bounding boxes and labels on the image
def draw_boxes(image, boxes, classes):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    print(boxes)
    for box, cls in zip(boxes, classes):
        # Draw rectangle
        box = [int(coordinate) for coordinate in box]
        draw.rectangle(((box[0], box[1]), (box[2], box[3])), outline="red", width=2)

    return image

# Set up the page title and header
st.title("Detectron2 Image Detection")
st.header("Upload an image to detect objects")

# Upload an image file
uploaded_file = st.file_uploader("Choose the document...", type=["pdf"])
if uploaded_file is None:
    st.write("Please upload the image in proper format.")
if uploaded_file is not None:
    # Display the uploaded image
    bytes_data = uploaded_file.read()
    image = convertPdfToImage(bytes_data)
    # image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Detecting...")

    # Convert the image to bytes
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='JPEG')
    img_bytes = img_bytes.getvalue()

    # Send the image to the Flask API
    response = requests.post("http://localhost:5000/predict", files={"file": img_bytes})

    if response.status_code == 200:
        results = response.json()
        boxes = results["boxes"]
        classes = results["classes"]
        # bbox_coords = results["bbox"]
        # cropped_image = None

        # if bbox_coords is not None:
        #     xmin = int(min(p[0] for p in bbox_coords))
        #     xmax = int(max(p[0] for p in bbox_coords))
        #     ymin = int(min(p[1] for p in bbox_coords))
        #     ymax = int(max(p[1] for p in bbox_coords))
        #     # xmax += 70
        #     # ymax += 40
        #     cropped_image = image[ymin:ymax, xmin:xmax]
        # Draw the boxes on the image
        
        image_with_boxes = draw_boxes(image, boxes, classes)

        # Display the image with bounding boxes
        # st.image(image, caption="cropped", use_column_width=True)
        st.image(image_with_boxes, caption="Processed Image with Detections.", use_column_width=True)
    else:
        st.write("Error in processing the image.")
