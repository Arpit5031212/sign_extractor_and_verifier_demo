import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
from paddleocr import PaddleOCR
import pypdfium2 as pdfium
import pandas as pd

# Define the custom cosine similarity layer
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

# Preprocess image function
def preProcessImage(imagepath, target_size=(224, 224)):
    image = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, target_size)
    image = np.reshape(image, (*target_size, 1))
    image = image / 255.0
    return image



# Load the trained model with custom layer
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("C:/Users/Prateek/Downloads/finalSignatureArebic23siamese_model2.keras", custom_objects={'CosineSimilarityLayer': CosineSimilarityLayer})

# Function to compare images and show similarity scores
def compare_with_user_images(user_id, path2, model, base_path="C:/Users/Prateek/Desktop/finalSignatureArebic/signatures/"):
    user_folder = os.path.join(base_path, user_id, f"{user_id}_original")
    if not os.path.isdir(user_folder):
        st.write(f"No folder found for user ID {user_id}.")
        return
    
    img2 = preProcessImage(path2)
    img2 = img2.reshape((1, 224, 224, 1))

    Similarity_scores = []
    for img_name in os.listdir(user_folder):
        img_path = os.path.join(user_folder, img_name)
        img1 = preProcessImage(img_path)
        img1 = img1.reshape((1, 224, 224, 1))

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

# Function to convert PDF to images
def convertPdfToImage(path):
    pdf = pdfium.PdfDocument(path)
    n_pages = len(pdf)
    images = []
    for page_number in range(n_pages):
        page = pdf.get_page(page_number)
        pil_image = page.render(
            scale=1,
            rotation=0,
            crop=(0, 0, 0, 0),
            grayscale=True
        )
        image = pil_image.to_pil()
        images.append(image)
    return images

# Initialize PaddleOCR
ocr = PaddleOCR(det_model_dir='C:/Users/Prateek/Downloads/en_PP-OCRv3_det_distill_train', rec_model_dir='C:/Users/Prateek/Downloads/en_PP-OCRv3_rec_train')

# Main Streamlit application
st.title("Signature Verification and Data Extraction System")

model = load_model()

uploaded_pdf = st.file_uploader("Choose the PDF file to extract signature and data from", type=["pdf"])


if uploaded_pdf:
    
    pdf_path = "temp_uploaded.pdf"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_pdf.getbuffer())

    # Extract signature
    empty_sign = "C:/Users/Prateek/Downloads/SPRN-ITU-0024050719130-1_rotated (1).pdf"
    extract_sign(pdf_path, empty_sign, "sign.png")
    extracted_img = cv2.imread("sign.png")

    image2 = Image.open("sign.png").convert('L')
    # st.image(uploaded_pdf , caption="uploaded pdf" , use_column_width= True)
    st.image(image2,caption='Extracted Signature Image', use_column_width=True)

    user_id = st.text_input("Enter the user ID:")
    
    # Convert PDF to images and perform OCR
    images = convertPdfToImage(pdf_path)
    data = []
    for page_number, image in enumerate(images):
        image = np.array(image)
        temp_image_path = f"temp_page_{page_number}.png"
        cv2.imwrite(temp_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        result = ocr.ocr(temp_image_path, cls=False)
        os.remove(temp_image_path)  # Clean up temp file

        # Extract and display data
        target_text = "certify"
        bbox_coords = None
        target_text_found = False

        for line in result:
            for word_info in line:
                box = word_info[0]  # Bounding box coordinates
                text = word_info[1][0]  # Extracted text
                text = str(text)
                if target_text in text:
                    bbox_coords = np.array(box, dtype=np.int32)
                    target_text_found = True
                    break
            if target_text_found:
                break

        if bbox_coords is not None:
            xmin = int(min(p[0] for p in bbox_coords))
            xmax = int(max(p[0] for p in bbox_coords))
            ymin = int(min(p[1] for p in bbox_coords))
            ymax = int(max(p[1] for p in bbox_coords))
            xmax += 70
            ymax += 40
            cropped_image = image[ymin:ymax, xmin:xmax]
            cropped_result = ocr.ocr(cropped_image, cls=False)
            cropped_text = " "
            avg_prediction = 0
            count = 0
            if cropped_result != [None]:
                for line in cropped_result:
                    for word_info in line:
                        text = word_info[1][0]
                        prediction = word_info[1][1]
                        cropped_text += text + " "
                        avg_prediction += prediction 
                        count += 1

            if avg_prediction != 0:
                avg_confidence = avg_prediction / count
            else:
                avg_confidence = 0

            if ':' in cropped_text:
                split_text = cropped_text.split(':')
                if len(split_text) > 1:
                    stripText = split_text[1].strip()
                    stripArray = stripText.split()
                    if len(stripArray) >= 3:
                        name = ' '.join(stripArray[:2])
                        address = ' '.join(stripArray[2:])
                        st.write(f"Extracted Name: {name}")
                        st.write(f"Extracted Address: {address}")
                        st.write(f"Avg Confidence Score: {avg_confidence:.4f}")
                        data.append({
                            "Name": name,
                            "Address": address,
                            "Avg Confidence Score": avg_confidence,
                            "pdf path" : pdf_path
                        })
                    else:
                        st.write("Not enough information found for name and address.")
                else:
                    st.write("Delimiter ':' not found in the text or no text after it.")
            else:
                st.write("Delimiter ':' not found in the text.")

    # Save the extracted data to a CSV
    # df = pd.DataFrame(data)
    # csv_output_path = os.path.join("output_directory", "extracted_data.csv")
    # df.to_csv(csv_output_path, index=False, encoding='utf-8')
    # st.write("Data saved to CSV.")

    if user_id:
        image2_path = "sign.png"
        if st.button("Show Prediction"):
            compare_with_user_images(user_id, image2_path, model)
