from flask import Flask, render_template, request, jsonify
import os
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import cv2

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load pre-trained model 
def load_model():
    model = models.efficientnet_b0(pretrained=True)
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize(300),  
    transforms.CenterCrop(256),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3)),  
])

# Dictionary mapping of common crops
CROP_CLASSES = {
    "corn": "corn",
    "rice": "rice",
    "wheat": "wheat",
    "tomato": "tomato",
    "potato": "potato",
    "soybean": "soybean"
}

# Function to analyze crop health based on more advanced color and texture features
def analyze_crop_health_advanced(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    height, width = gray_img.shape
    lbp_img = np.zeros_like(gray_img, dtype=np.uint8)

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            center = gray_img[y, x]
            binary_pattern = 0
            for i in range(8):
                offset_x = int(np.round(np.cos(2 * np.pi * i / 8)))
                offset_y = int(np.round(np.sin(2 * np.pi * i / 8)))
                neighbor_x = x + offset_x
                neighbor_y = y + offset_y

                if 0 <= neighbor_x < width and 0 <= neighbor_y < height:
                    if gray_img[neighbor_y, neighbor_x] >= center:
                        binary_pattern |= (1 << i)

            lbp_img[y, x] = binary_pattern

    # Calculate LBP histogram
    lbp_hist = cv2.calcHist([lbp_img], [0], None, [256], [0, 256])
    lbp_hist_normalized = lbp_hist.astype("float") / (lbp_img.size + 1e-6)
    texture_complexity = np.std(lbp_hist_normalized)

    health_score = 50  # Base score

    # --- Color Analysis ---
    hsv_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    green_mask = cv2.inRange(hsv_img, (35, 40, 40), (85, 255, 255))
    green_percentage = np.sum(green_mask > 0) / (green_mask.shape[0] * green_mask.shape[1])
    yellow_mask = cv2.inRange(hsv_img, (20, 100, 100), (35, 255, 255))
    yellow_percentage = np.sum(yellow_mask > 0) / (yellow_mask.shape[0] * yellow_mask.shape[1])

    if green_percentage > 0.6:
        health_score += 30
    elif green_percentage > 0.4:
        health_score += 20
    elif green_percentage < 0.2:
        health_score -= 30

    if yellow_percentage > 0.2:
        health_score -= 25

    # --- Texture Complexity Influence ---
    if texture_complexity > 0.05:
        health_score += 10
    elif texture_complexity < 0.02:
        health_score -= 15

    health_score = np.clip(health_score, 0, 100)

    if health_score > 80:
        health_status = "Excellent"
    elif health_score > 60:
        health_status = "Good"
    elif health_score > 40:
        health_status = "Fair"
    else:
        health_status = "Poor"

    return {
        "status": health_status,
        "score": int(health_score),
        "green_percentage": round(green_percentage * 100, 1),
        "yellow_percentage": round(yellow_percentage * 100, 1),
        "texture_complexity": round(texture_complexity, 4)
    }

def generate_recommendations_advanced(crop_type, health_data):
    recommendations = []
    health_status = health_data["status"]
    yellow_percentage = health_data["yellow_percentage"]
    green_percentage = health_data["green_percentage"]
    score = health_data["score"]

    if health_status in ["Poor", "Very Poor"]:
        recommendations.append(f"The analysis indicates {crop_type} is exhibiting signs of significant stress. Immediate attention is advised to diagnose the underlying cause.")
        if yellow_percentage > 15:
            recommendations.append(f"Noticeable yellowing suggests potential nutrient deficiencies, possibly nitrogen, potassium, or magnesium. Consider a soil test for precise diagnosis and targeted fertilization with an NPK blend.")
        if green_percentage < 30:
            recommendations.append(f"Low green intensity could point to issues like insufficient chlorophyll production due to inadequate light, water stress, or disease. Investigate environmental conditions and check for any signs of pests or pathogens.")
        recommendations.append("Monitor closely for any further deterioration and consult with an agricultural expert if the condition persists or worsens.")

    elif health_status == "Fair":
        recommendations.append(f"The {crop_type} shows moderate health. Consistent monitoring is recommended to prevent potential decline.")
        if yellow_percentage > 10:
            recommendations.append(f"Slight yellowing might indicate an early stage of nutrient imbalance. A balanced fertilizer application could be beneficial. Consider foliar feeding for quicker absorption.")
        if green_percentage < 45:
            recommendations.append(f"Slightly lower green levels could be due to various factors. Ensure optimal watering and light exposure. Regular checks for early signs of pests or diseases are crucial.")

    elif health_status == "Good":
        recommendations.append(f"The {crop_type} appears to be in good health. Maintain current cultivation practices.")
        recommendations.append("Continue regular scouting for any early indicators of stress or disease to ensure sustained healthy growth.")

    elif health_status == "Excellent":
        recommendations.append(f"The {crop_type} is in excellent condition, indicating optimal growth. Continue your successful management strategies.")
        recommendations.append("Periodic monitoring for any subtle changes and adapt practices proactively.")

    if crop_type == "corn":
        if score < 50:
            recommendations.append("Assess soil nitrogen levels critically; corn is a heavy nitrogen feeder, especially during vegetative growth. Consider side-dressing with urea or ammonium nitrate.")
        elif yellow_percentage > 12:
            recommendations.append("Evaluate for potential magnesium or sulfur deficiencies, which can manifest as interveinal yellowing. Soil testing can confirm these imbalances.")
        recommendations.append("Monitor for common corn pests like fall armyworm and corn borer, especially during vulnerable growth stages.")
    elif crop_type == "rice":
        if score < 60:
            recommendations.append("Ensure consistent water management, crucial for rice paddy health. Check for adequate flood depth and drainage as needed.")
        recommendations.append("Be vigilant for rice blast and sheath blight, particularly in warm and humid conditions. Consider preventative fungicide applications if these diseases are prevalent in your region.")
        if green_percentage < 55:
            recommendations.append("Investigate potential iron deficiency (Khaira disease), often seen in alkaline soils. Foliar sprays of ferrous sulfate may provide temporary relief.")
    elif crop_type == "wheat":
        if score < 45:
            recommendations.append("Inspect for signs of fungal diseases such as rusts and powdery mildew, which can rapidly spread in wheat crops. Timely fungicide applications are often necessary.")
        if yellow_percentage > 8:
            recommendations.append("Assess soil health and nutrient availability, especially nitrogen, which is critical for tillering and grain fill in wheat.")
        recommendations.append("Monitor for aphid infestations, which can transmit viral diseases and reduce yield.")
    elif crop_type == "tomato":
        if score < 55:
            recommendations.append("Examine leaves for symptoms of early blight, late blight, or other fungal diseases common in tomatoes. Ensure good air circulation and consider protective fungicide sprays.")
        if yellow_percentage > 10:
            recommendations.append("Check for blossom end rot, often linked to calcium deficiency or inconsistent watering. Ensure adequate calcium supply and consistent soil moisture.")
        recommendations.append("Scout for common tomato pests like tomato hornworms and whiteflies, and implement integrated pest management strategies.")
    elif crop_type == "potato":
        if score < 65:
            recommendations.append("Monitor for early and late blight, serious fungal diseases in potatoes. Protective fungicide applications are crucial, especially in wet weather.")
        recommendations.append("Ensure consistent soil moisture to prevent common scab and promote uniform tuber development.")
        if green_percentage < 60:
            recommendations.append("Assess nitrogen and potassium levels, essential for vegetative growth and tuber formation in potatoes.")
    elif crop_type == "soybean":
        if score < 50:
            recommendations.append("Check for signs of soybean cyst nematode (SCN), a major yield-reducing pest. Consider resistant varieties and crop rotation in subsequent seasons.")
        recommendations.append("Be aware of fungal diseases like soybean rust and downy mildew, especially in humid conditions. Foliar fungicides may be necessary.")
        if yellow_percentage > 10:
            recommendations.append("Evaluate for manganese deficiency, which can occur in high pH soils. Foliar application of manganese sulfate may be beneficial.")

    return recommendations

# Load the model at startup
model = load_model()

@app.route('/')
def index():
    return render_template('index.html', crops=sorted(CROP_CLASSES.keys()))

@app.route('/analyze', methods=['POST'])
def analyze_crop():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    crop_type = request.form.get('crop_type')
    if not crop_type or crop_type not in CROP_CLASSES:
        return jsonify({'error': 'Please select a valid crop type'})

    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        try:
            # Analyze crop health
            health_data = analyze_crop_health_advanced(filename)

            # Generate recommendations
            recommendations = generate_recommendations_advanced(crop_type, health_data)

            return jsonify({
                'filename': file.filename,
                'crop_type': crop_type,
                'health_status': health_data['status'],
                'health_score': health_data['score'],
                'recommendations': recommendations
            })

        except Exception as e:
            return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)