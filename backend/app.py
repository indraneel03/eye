from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import base64
import cv2
import numpy as np
from timm.models import create_model
import os
import logging
import traceback
import shap
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": ["http://localhost:3000"], "methods": ["GET", "POST", "OPTIONS"], "allow_headers": ["Content-Type"]}})

# Create uploads directory if it doesn't exist
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Model class
class TransferLearningModel(nn.Module):
    def __init__(self, model_name, num_classes=4):
        super(TransferLearningModel, self).__init__()

        if model_name == 'mobilenet_v3':
            self.model = models.mobilenet_v3_small(pretrained=False)
            self.model.features[0][0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
            num_features = self.model.classifier[3].in_features
            self.model.classifier[3] = nn.Linear(num_features, num_classes)

        elif model_name == 'efficientnet_b0':
            self.model = models.efficientnet_b0(pretrained=False)
            self.model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
            num_features = self.model.classifier[1].in_features
            self.model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(num_features, num_classes))

        elif model_name == 'squeezenet1_1':
            self.model = models.squeezenet1_1(pretrained=False)
            self.model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1, bias=False)
            num_features = self.model.classifier[1].in_channels
            self.model.classifier[1] = nn.Conv2d(num_features, num_classes, kernel_size=1)

        elif model_name == 'resnet50':
            self.model = models.resnet50(pretrained=False)
            self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, num_classes)

        elif model_name == 'resnet18':
            self.model = models.resnet18(pretrained=False)
            self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, num_classes)

        elif model_name == 'mobilevit_s':
            self.model = create_model('mobilevit_s', pretrained=False, num_classes=num_classes)
            self.model.stem.conv = nn.Conv2d(1, self.model.stem.conv.out_channels, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x):
        return self.model(x)

# Cache for loaded models
model_cache = {}

def load_model(model_name):
    try:
        if model_name not in model_cache:
            logger.info(f"Loading model: {model_name}")
            model_path = os.path.join('models', f'best_{model_name}_model.pth')

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")

            if model_name == 'mobilevit_s':
                model = create_model('mobilevit_s', pretrained=False, num_classes=4)
                model.stem.conv = nn.Conv2d(1, model.stem.conv.out_channels, kernel_size=3, stride=2, padding=1, bias=False)
            else:
                model = TransferLearningModel(model_name)

            checkpoint = torch.load(model_path, map_location=device)

            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)

            model.to(device)
            model.eval()
            model_cache[model_name] = model
            logger.info(f"Model {model_name} loaded successfully")

        return model_cache[model_name]
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {str(e)}")
        raise

def preprocess_image(image):
    try:
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        return transform(image).unsqueeze(0)
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

def get_target_layer(model, model_name):
    if model_name == 'mobilenet_v3':
        return model.model.features[-1]
    elif model_name == 'efficientnet_b0':
        return model.model.features[-1]
    elif model_name == 'squeezenet1_1':
        return model.model.features[-1]
    elif model_name == 'resnet50':
        return model.model.layer4
    elif model_name == 'resnet18':
        return model.model.layer4
    elif model_name == 'mobilevit_s':
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d)):
                return module
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")

def generate_gradcam(model, input_tensor, model_name, target_class=None):
    try:
        gradients = None
        activations = None

        target_layer = get_target_layer(model, model_name)
        
        def backward_hook(module, grad_in, grad_out):
            nonlocal gradients
            gradients = grad_out[0]

        def forward_hook(module, input, output):
            nonlocal activations
            activations = output

        handle1 = target_layer.register_forward_hook(forward_hook)
        handle2 = target_layer.register_backward_hook(backward_hook)

        output = model(input_tensor)
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()

        model.zero_grad()
        score = output[:, target_class]
        score.backward()

        with torch.no_grad():
            weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * activations, dim=1).squeeze(0)
            cam = torch.relu(cam)
            cam = cam.cpu().numpy()

        handle1.remove()
        handle2.remove()

        cam = cv2.resize(cam, (input_tensor.size(3), input_tensor.size(2)))
        
        # Normalize the CAM
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-7)
        
        # Apply color mapping
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        
        # Convert BGR to RGB
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        return heatmap
    
    except Exception as e:
        logger.error(f"Error generating GradCAM: {str(e)}")
        raise


@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Server is running'}), 200

@app.route('/api/classify', methods=['POST'])
def classify_image():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        model_name = request.form.get('model', 'mobilenet_v3')

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Save file temporarily
        temp_path = os.path.join('uploads', file.filename)
        file.save(temp_path)

        # Load and preprocess image
        image = Image.open(temp_path).convert('RGB')
        input_tensor = preprocess_image(image).to(device)

        # Load model and perform inference
        model = load_model(model_name)
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            # Modified to show two decimal points
            confidence = round(confidence.item() , 4)

        # Generate Grad-CAM
        heatmap = generate_gradcam(model, input_tensor, model_name, predicted.item())

        # Convert images to base64
        _, img_encoded = cv2.imencode('.png', cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')

        _, heatmap_encoded = cv2.imencode('.png', heatmap)
        heatmap_base64 = base64.b64encode(heatmap_encoded).decode('utf-8')

        # Clean up
        os.remove(temp_path)

        classes = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
        return jsonify({
            'class': classes[predicted.item()],
            'confidence': confidence,
            'gradcam': heatmap_base64,
            'original': img_base64
        })

    except Exception as e:
        logger.error(f"Error in classify_image: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    # Try different ports
    ports = [5000, 8000, 8080, 3001]
    
    for port in ports:
        try:
            logger.info(f"Attempting to start server on port {port}")
            app.run(host='0.0.0.0', port=port, debug=True)
            break
        except OSError as e:
            logger.error(f"Port {port} is in use, trying next port...")
            continue
