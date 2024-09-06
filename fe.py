import streamlit as st
import requests
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageOps
import io
import pypdfium2 as pdfium
from app import get_pensioner_sign_area, get_target_text
import tensorflow as tf
import numpy as np
import cv2

import os

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

def enhance_image(image, scale_factor=2):
    width, height = image.size
    enhanced_image = image.resize((width * scale_factor, height * scale_factor), Image.LANCZOS)
    return enhanced_image


def draw_boxes(image, boxes, classes):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for box, cls in zip(boxes, classes):
        box = [int(coordinate) for coordinate in box]
        # Draw the rectangle
        draw.rectangle(((box[0], box[1]), (box[2], box[3])), outline="red", width=2)
        
    return image

def draw_boxes_and_crop(image, boxes, classes, padding=2):
    # Draw boxes on a copy of the original image
    image_with_boxes = image.copy()
    image_with_boxes = draw_boxes(image_with_boxes, boxes, classes)
    
    cropped_regions = []
    
    for box, cls in zip(boxes, classes):
        box = [int(coordinate) for coordinate in box]

          # Add padding to the bounding box
        box_with_padding = [
            max(0, box[0] - padding),  # Left
            max(0, box[1] - padding),  # Top
            min(image.width, box[2] + padding),  # Right
            min(image.height, box[3] + padding)  # Bottom
        ]
        
        # Crop the region based on the bounding box
        cropped_region = image.crop((box_with_padding[0], box_with_padding[1], box_with_padding[2], box_with_padding[3]))
        
        # Enhance the cropped image quality
        cropped_region = enhance_image(cropped_region, scale_factor=2)  # Adjust scale factor as needed
        cropped_regions.append(cropped_region)

    return image_with_boxes, cropped_regions

def apply_final_processing(image):
    # Apply additional enhancement to the selected image
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(3)  # Increase contrast
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(3)  # Increase sharpness
    enhancer= ImageEnhance.Brightness(image)
    image = enhancer.enhance(1)
    # Add padding and margin
    # padding = 20
    # image = ImageOps.expand(image, fill='white')
    
    return image
class CosineSimilarityLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CosineSimilarityLayer, self).__init__(**kwargs)

    def call(self, inputs):
        vector1, vector2 = inputs
        norm_vector1 = tf.sqrt(tf.reduce_sum(tf.square(vector1), axis=1, keepdims=True))
        norm_vector2 = tf.sqrt(tf.reduce_sum(tf.square(vector2), axis=1, keepdims=True))
        dot_product = tf.reduce_sum(vector1 * vector2, axis=1, keepdims=True)
        cosine_sim = dot_product / (norm_vector1 * norm_vector2 + tf.keras.backend.epsilon())
        return cosine_sim

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 1)

# def preProcessImage(imagepath, target_size=(224, 224)):
#     image = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)
#     image = cv2.resize(image, target_size)
#     image = np.reshape(image, (*target_size, 1))
#     image = image / 255.0
#     return image
import cv2
import numpy as np

def preProcessImage(imagepath, target_size=(224, 224)):
    # Read the image in grayscale
    image = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)
    
    # Apply Gaussian blur to reduce noise
    image = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Apply adaptive thresholding to enhance clarity and make background white
    image = cv2.adaptiveThreshold(
        image, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        11, 
        2
    )
    
    # Resize the image to the target size
    image = cv2.resize(image, target_size)
    
    # Add a channel dimension (1 channel for grayscale)
    image = np.reshape(image, (*target_size, 1))
    
    # Normalize the image
    image = image / 255.0
    
    return image


@st.cache_resource
def load_model():    
    st.write("model is loading")
    return tf.keras.models.load_model("C:/Users/Arpit/Downloads/finalSignatureArebic23siamese_model2.keras", custom_objects={'CosineSimilarityLayer': CosineSimilarityLayer})

def compare_with_user_images(user_id, path2, model, base_path="C:/Users/Arpit/Downloads/signatureszip/signatures"):
   
    user_folder = os.path.join(base_path, user_id, f"{user_id}_original")
    if not os.path.isdir(user_folder):
        st.write(f"No folder found for user ID {user_id}.")
        return
    
    img2 = preProcessImage(path2)
    img2 = img2.reshape((1, 224, 224, 1))

    for img_name in os.listdir(user_folder):
        img_path = os.path.join(user_folder, img_name)
        img1 = preProcessImage(img_path)
        img1 = img1.reshape((1, 224, 224, 1))

        Similarity_scores = []

        prediction = model.predict([img1, img2])
        similarity_score = prediction[0][0]
        Similarity_scores.append(similarity_score)
        # st.write(Similarity_scores)


    if Similarity_scores:
        max_score = max(Similarity_scores)
        min_score = min(Similarity_scores)
        avg_score = sum(Similarity_scores) / len(Similarity_scores)

        st.write(f"Maximum similarity score: {max_score:.4f}")
        st.write(f"Minimum similarity score: {min_score:.4f}")
        st.write(f"Average similarity score: {avg_score:.4f}")


model = load_model()
# Set up the page title and header
st.title("Detectron2 Image Detection")
user_id = st.text_input("Enter the user ID:")
# user_id = st.text_input("Enter the user ID:")
st.header("Upload a PDF document to detect objects")

# Upload a PDF file

uploaded_file = st.file_uploader("Choose the document...", type=["pdf"])
empty_sign = "C:/Users/Arpit/Downloads/SPRN-ITU-0024050719130-1_rotated (1).pdf"
if uploaded_file is  None:
    st.write("Please upload the PDF document.")
if uploaded_file is not None:
    # Display the uploaded PDF as an image
    bytes_data = uploaded_file.read()
    image = convertPdfToImage(bytes_data)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Detecting...")
    
    image = np.array(image)
    pensioner_sign_area, roi = get_pensioner_sign_area(image)
    print(pensioner_sign_area)
    image = Image.fromarray(pensioner_sign_area)

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
        # cropped_region = results["roi"]
        # print("roi", cropped_region)
        # Draw the boxes and crop the regions
        image_with_boxes, cropped_images = draw_boxes_and_crop(image, boxes, classes)

        # Display the processed image with bounding boxes
        st.image(image_with_boxes, caption="Processed Image with Detections.", use_column_width=True)

        # Display cropped images in a grid
        n_images = len(cropped_images)
        cols = st.columns(3)  # Adjust number of columns as needed

        selected_image = None
        selected_index = None
        
        image_1 = np.array(image)
        for i, img in enumerate(cropped_images):
            with cols[i % len(cols)]:  # Ensure images are distributed across columns
                st.image(img, caption=f"Cropped Region {i+1}", use_column_width=True)
                if st.button(f"Select Region {i+1}", key=i):
                    selected_image = img
                    selected_index = i

        if selected_image:
            final_image = apply_final_processing(selected_image)
    
            # Save the final image to a BytesIO object
            final_image_io = io.BytesIO()
            final_image.save(final_image_io, format='PNG')
            final_image_io.seek(0)  # Go to the beginning of the BytesIO object
            
            st.image(final_image, caption="Selected Cropped Region", use_column_width=True)

            

            if user_id:
                
                
        
                # Open the image from the BytesIO object
                image2 = Image.open(final_image_io).convert('L')
                # st.image(image2, caption='Uploaded Image.', use_column_width=True)

                # Save the image to a temporary file
                image2_path = "temp_image2.png"
                image2.save(image2_path)
                compare_with_user_images(user_id, image2_path, model)      
    else:
        st.write("Error in processing the image.")