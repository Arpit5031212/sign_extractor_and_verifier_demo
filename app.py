from flask import Flask, request, jsonify
from PIL import Image, ImageEnhance
import numpy as np
import io
from signature_extractor import *
from paddleocr import PaddleOCR
import re

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


app = Flask(__name__)
detector = Detector()
ocr = PaddleOCR(det_model_dir='C:/Users/Arpit/Downloads/en_PP-OCRv3_det_distill_train', rec_model_dir='C:/Users/Arpit/Downloads/en_PP-OCRv3_rec_train', use_gpu=False)


def merge_signature_boxes(signature_boxes):
    def is_inside(box1, box2):
        return (box1[0] >= box2[0] and box1[2] <= box2[2] and
                box1[1] >= box2[1] and box1[3] <= box2[3])

    def merge_boxes(boxes):
        min_x = min(box[0] for box in boxes)
        min_y = min(box[1] for box in boxes)
        max_x = max(box[2] for box in boxes)
        max_y = max(box[3] for box in boxes)
        return [min_x, min_y, max_x, max_y]

    # Process the boxes
    if len(signature_boxes) > 1:
        # Check if one box is inside the other
        outside_box = None
        for i in range(len(signature_boxes)):
            for j in range(i+1, len(signature_boxes)):
                if is_inside(signature_boxes[i], signature_boxes[j]):
                    outside_box = signature_boxes[j]
                elif is_inside(signature_boxes[j], signature_boxes[i]):
                    outside_box = signature_boxes[i]

        # If no box is inside another, merge the boxes on the same horizontal axis
        if outside_box is None:
            outside_box = merge_boxes(signature_boxes)
    else:
        outside_box = signature_boxes[0]

    return outside_box

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

def get_target_text(image):
    fallback = False
    target_texts_1 = [
        'affixed',  # Correct spelling
        'affix',    # Base word
        'affixe',   # Common misspelling or partial word
        'afixed',   # Common typo (missing 'f')
        'affxd',    # Another variant that could arise from OCR errors
    ]
    
    target_texts_2 = [
        '(Signature of Pensioner)',
        'Signature of Pensioner',
        'Signature of Pensionr',
        '(Signature of Penager)',
        'Signature of Pensiner',
        'Signature of Pesoner',
        'Signature of Pesonr',
    ]
    
    target_texts_3 ="certify"

    result = ocr.ocr(image, cls=False)
    target_text_found = False
    bbox_coords = None
    bbox_coords2 = None
    
    target_text_pattern_1 = re.compile(r'({})'.format('|'.join(target_texts_1)), re.IGNORECASE)
    target_text_pattern_2 = re.compile(r'({})'.format('|'.join(target_texts_2)), re.IGNORECASE)
    
    for line in result:
        for word_info in line:
            box = word_info[0]
            text = word_info[1][0]
            text = str(text)
            # Use regex to search for any of the target texts
            if re.search(target_text_pattern_1, text):
                bbox_coords = np.array(box, dtype=np.int32)
                target_text_found = True
                break
            if target_text_found:
                break
            
    for line in result:
        for word_info in line:
            box = word_info[0]
            text = word_info[1][0]
            text = str(text)
            # Use regex to search for any of the target texts
            if re.search(target_text_pattern_2, text):
                bbox_coords = np.array(box, dtype=np.int32)
                target_text_found = True
                fallback = True
                break
            if target_text_found:
                break
    
    for line in result:
        for word_info in line:
            box = word_info[0]  # Bounding box coordinates
            text = word_info[1][0]  # Extracted text
            # Ensure text is a string
            text = str(text)
            if target_texts_3 in text:
                bbox_coords2 = np.array(box, dtype=np.int32)
                target_text_found = True
                break
        if target_text_found:
            break
    
    pensioner_number_bbox = None
    if bbox_coords2 is not None:
        # Crop the bounding box area from the image
        xmin = int(min(p[0] for p in bbox_coords2)) + 83
        xmax = int(max(p[0] for p in bbox_coords2)) + 10
        ymin = int(min(p[1] for p in bbox_coords2)) - 20
        ymax = int(max(p[1] for p in bbox_coords2)) - 20
        xmax += 70
        ymax += 40
         
        pensioner_number_bbox = [xmin, ymin, xmax, ymax - 40]
    
    cropped_image = image[ymin:ymax, xmin:xmax]
    cropped_result = ocr.ocr(cropped_image, cls=False)
    # Initialize variables
    reference_number = ""
    # Check index ranges and extract data if available
    if cropped_result:
        if len(cropped_result) > 0:
            if cropped_result[0] is not None and len(cropped_result[0]) > 0:
                reference_number = cropped_result[0][0][1][0]
                
    
    return bbox_coords, reference_number, pensioner_number_bbox, fallback

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

def get_pensioner_sign_area(image):
    
    affixed_bbox = None
    reference_number = None
    reference_number_bbox = None
    fallback = False
    roi = None
    masked_image = None
    
    affixed_bbox, reference_number, reference_number_bbox, fallback = get_target_text(image)
    
    xmin = None
    xmax = None
    ymin = None
    ymax = None
    
    if affixed_bbox is not None:
        if fallback is False:
            xmin = affixed_bbox[2][0]
            xmax = image.shape[1]
            ymin = 0
            ymax = affixed_bbox[2][1] + 30
        else:
            xmin = affixed_bbox[0][0] - 50
            xmax = image.shape[1]
            ymin = 0
            ymax = affixed_bbox[2][1]

        # Create a mask with the same shape as the image, initially all black (0)
        mask = np.zeros_like(image, dtype=np.uint8)
        
        # Fill the ROI area in the mask with white (255 for grayscale, or 1 for binary mask)
        mask[ymin:ymax, xmin:xmax] = 255
        
        # Apply the mask to the original image (bitwise AND to keep only the ROI)
        masked_image = cv2.bitwise_and(image, mask)
            
        roi = [xmin, xmax, ymin, ymax]
    
    return masked_image, roi, reference_number, reference_number_bbox

    

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files['file']
    image = Image.open(io.BytesIO(file.read())).convert(mode='RGB')
    
    # image = preProcessImage(image)
    # image = apply_final_processing(image)
    image = np.array(image)
    outputs = detector.onImage(image)
    viz = Visualizer(image[:,:,::-1], metadata = metadata, instance_mode = ColorMode.IMAGE_BW)
    output = viz.draw_instance_predictions(outputs["instances"].to("cpu"))
    
    pred_boxes = outputs['instances'].pred_boxes.tensor.tolist()
    pred_classes = outputs['instances'].pred_classes.numpy()
    
    merge_boxes = merge_signature_boxes(pred_boxes)
    results = {
        "boxes": merge_boxes,
        "classes": pred_classes.tolist(),
    }
    return jsonify(results)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)