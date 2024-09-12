import streamlit as st
from streamlit_cropper import st_cropper
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
            grayscale=False
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
    draw.rectangle(((boxes[0], boxes[1]), (boxes[2], boxes[3])), outline="red", width=2) 
    return image

def draw_boxes_and_crop(image, boxes, classes, padding=2):
    # Draw boxes on a copy of the original image
    image_with_boxes = image.copy()
    image_with_boxes = draw_boxes(image_with_boxes, boxes, classes)
    cropped_region = image.crop((boxes[0], boxes[1], boxes[2], boxes[3]))
    # Enhance the cropped image quality
    cropped_region = enhance_image(cropped_region, scale_factor=2)  # Adjust scale factor as needed

    return image_with_boxes, cropped_region

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

def preProcessImage(image, target_size=(224, 224)):
    
    image = ImageOps.grayscale(image)
    image = np.array(image)
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

def compare_with_user_images(user_id, image, model, base_path="D:/final_extraction_model/output_folder"):
    
    user_folder = os.path.join(base_path, f"{user_id}")
    print(user_folder)
    # Check if the folder exists
    print(os.path.isdir(user_folder))
    if not os.path.isdir(user_folder):
        st.write(f"No folder found for user ID {user_id}.")
        # Ask the user if they want to create a new folder
        create_new_folder = st.radio(f"Do you want to create a new folder for user ID {user_id}?", ('No', 'Yes'))
        
        if create_new_folder == 'Yes':
            os.makedirs(user_folder)
            st.write(f"New folder created for user ID {user_id}.")
            # Save the cropped image in the newly created folder
            cropped_image_path = os.path.join(user_folder, f"{user_id}_signature.jpg")
            image.save(cropped_image_path)
            st.write(f"Signature saved at: {cropped_image_path}")
        else:
            st.write("No new folder created. Please enter a valid user ID.")
        return
    
    img2 = preProcessImage(image)
    img2 = img2.reshape((1, 224, 224, 1))
    
    for img_name in os.listdir(user_folder):
        img_path = os.path.join(user_folder, img_name)
        base_image = Image.open(img_path)
        img1 = preProcessImage(base_image)
        img1 = img1.reshape((1, 224, 224, 1))
        Similarity_scores = []
        prediction = model.predict([img1, img2])
        similarity_score = prediction[0][0]
        Similarity_scores.append(similarity_score)
        
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
    pensioner_sign_area, roi, reference_number = get_pensioner_sign_area(image)
    
    if reference_number and reference_number.startswith("REF"):
        
        reference_number = reference_number.replace("REF", "")
        reference_number = reference_number.replace("#", "")
        reference_number = reference_number.replace(".", "")
        reference_number = reference_number.replace("=", "")
        reference_number = reference_number.replace(",", "")
        
        cleaned_reference = reference_number
        
        user_id = cleaned_reference.split('/')[0]  # Extract '460972'
        st.write(f"Detected Reference Number: {reference_number}")
        st.write(f"Extracted User ID: {user_id}")
    else:
        st.write("Unable to detect a valid reference number.")
        user_id = st.text_input("Please enter the user ID manually:")
    
    if pensioner_sign_area is not None:
        
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
            extracted_signature = None
                
            image_with_boxes, cropped_image = draw_boxes_and_crop(image, boxes, classes)

            # Display the processed image with bounding boxes
            st.image(image_with_boxes, caption="Processed Image with Detections.", use_column_width=True)

            extracted_signature = apply_final_processing(cropped_image)
            
            st.image(extracted_signature, caption="Selected Cropped Region", use_column_width=True)

            if user_id:
                compare_with_user_images(user_id, extracted_signature, model)
        else:
            st.write("Error in detecting the signature. Please crop the reason containing signature.")
            # Get a cropped image from the frontend
            cropped_img = st_cropper(image, realtime_update=True, box_color='#000FFF', aspect_ratio=None)
            print(cropped_img)
            st.image(cropped_img, caption='cropped by user')
            if user_id:
                compare_with_user_images(user_id, cropped_img, model)
    else:
        st.write("Error in processing the image. Please Crop the image for further processing.")
        cropped_img = st_cropper(image, realtime_update=True, box_color='#000FFF', aspect_ratio=None)
        print(cropped_img)
        st.image(cropped_img, caption='cropped by user')
        if user_id:
            compare_with_user_images(user_id, cropped_img, model)
        