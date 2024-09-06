from flask import Flask, request, jsonify
from PIL import Image, ImageEnhance
import numpy as np
import io
from signature_extractor import *
from paddleocr import PaddleOCR
app = Flask(__name__)
detector = Detector()
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
ocr = PaddleOCR(det_model_dir='C:/Users/Arpit/Downloads/en_PP-OCRv3_det_distill_train', rec_model_dir='C:/Users/Arpit/Downloads/en_PP-OCRv3_rec_train', use_gpu=False)

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

def get_target_text(image, target_text):
    result = ocr.ocr(image, cls=False)
    target_text_found = False
    bbox_coords = None
    cropped_image = None
    for line in result:
        for word_info in line:
            box = word_info[0]
            text = word_info[1][0]
            text = str(text)
            if target_text in text:
                bbox_coords = np.array(box, dtype=np.int32)
                target_text_found = True
                break
            if target_text_found:
                break

            if bbox_coords is not None:
                # Crop the bounding box area from the image
                xmin = int(min(p[0] for p in bbox_coords)) + 83
                xmax = int(max(p[0] for p in bbox_coords)) + 0
                ymin = int(min(p[1] for p in bbox_coords)) - 20
                ymax = int(max(p[1] for p in bbox_coords)) - 20

                xmax += 70
                ymax += 40
    return bbox_coords

def preProcessImage(image, target_size=(224, 224)):
    
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
    
def extract_signature(image, target_text):
        result = ocr.ocr(image, cls=False)
        target_text_found = False
        bbox_coords = None
        cropped_image = None
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
        return cropped_image, bbox_coords

def get_pensioner_sign_area(image):
    affixed_bbox = get_target_text(image, "affixed")
        
    xmin = affixed_bbox[2][0]
    xmax = image.shape[1]
    ymin = 0
    ymax = affixed_bbox[2][1] + 50
    
    roi = [xmin, xmax, ymin, ymax]
    cropped = image[ymin:ymax, xmin:xmax]
    return cropped, roi

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files['file']
    image = Image.open(io.BytesIO(file.read())).convert(mode='RGB')
    
    # image = preProcessImage(image)
    # image = apply_final_processing(image)
    image = np.array(image)
    print("image shape: ", image.shape)
    outputs = detector.onImage(image)
    viz = Visualizer(image[:,:,::-1], metadata = metadata, instance_mode = ColorMode.IMAGE_BW)
    output = viz.draw_instance_predictions(outputs["instances"].to("cpu"))
    # Process the outputs and format as needed
    # instances = outputs["instances"].to("cpu")
    pred_boxes = outputs['instances'].pred_boxes.tensor.tolist()
    pred_classes = outputs['instances'].pred_classes.numpy()
    results = {
        "boxes": pred_boxes,
        "classes": pred_classes.tolist(),
        
        # "cropped_box": bbox,
        # "roi": roi
        # Add more fields as needed
    }
    return jsonify(results)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)