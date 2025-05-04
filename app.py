from flask import Flask, render_template, request, jsonify
import os
import torch
from PIL import Image
import numpy as np
import cv2
from models.model import EnhancedUNetWithSE

app = Flask(__name__)

# Load the pre-trained model
model = EnhancedUNetWithSE()
model.load_state_dict(torch.load("models/enhanced_model.pth", map_location=torch.device('cpu')))
model.eval()

# Define the image enhancement function
def enhance_image(img):
    img = img.resize((256, 256))  # Resize to match model input
    img_tensor = torch.tensor(np.array(img).transpose((2, 0, 1)), dtype=torch.float32) / 255.0
    img_tensor = img_tensor.unsqueeze(0)

    # Perform enhancement
    with torch.no_grad():
        enhanced_image_tensor = model(img_tensor)

    # Convert to numpy array and save
    enhanced_image = enhanced_image_tensor.squeeze().permute(1, 2, 0).numpy() * 255.0
    enhanced_image = np.clip(enhanced_image, 0, 255).astype(np.uint8)

    # Save enhanced image
    enhanced_image_path = os.path.join("static/enhanced", "enhanced_image.png")
    cv2.imwrite(enhanced_image_path, cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR))

    return enhanced_image_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/enhance', methods=['POST'])
def enhance():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    image_file = request.files['image']
    image_path = os.path.join("static/upload", image_file.filename)
    image_file.save(image_path)

    # Load and enhance the image
    img = Image.open(image_path).convert("RGB")
    enhanced_image_path = enhance_image(img)

    # Return the original and enhanced image paths
    return jsonify({
        'original_url': f'/static/upload/{image_file.filename}',
        'image_url': enhanced_image_path
    })

if __name__ == "__main__":
    app.run(debug=True)
