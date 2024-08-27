from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import io
from signature_extractor import *
from paddleocr import PaddleOCR

app = Flask(__name__)
detector = Detector()

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
ocr = PaddleOCR(use_angle_cls=True, use_gpu = False)

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

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    image = np.array(image)
    # image, bbox = extract_signature(image, "affixed")
    
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
        # "cropped_image": image
        # Add more fields as needed
    }
    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
