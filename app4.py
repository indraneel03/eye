import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from timm.models import create_model
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

        elif model_name == 'mobilevit':
            # Load the base model without modifications first
            self.model = create_model('mobilevit_s', pretrained=False, num_classes=num_classes)
            # Then modify the first conv layer
            self.model.stem.conv = nn.Conv2d(1, self.model.stem.conv.out_channels, 
                                           kernel_size=3, stride=2, padding=1, bias=False)
    
    def forward(self, x):
        return self.model(x)

# Cache the model loading function
@st.cache_resource
def load_model(model_name):
    if model_name == 'mobilevit':
        # For MobileViT, load the raw state dict first
        model = create_model('mobilevit_s', pretrained=False, num_classes=4)
        # Modify the first conv layer after loading the base model
        model.stem.conv = nn.Conv2d(1, model.stem.conv.out_channels, 
                                  kernel_size=3, stride=2, padding=1, bias=False)
        # Load the state dict
        checkpoint = torch.load('best_mobilevit_s_model.pth', map_location=device)
        model.load_state_dict(checkpoint)
    else:
        # For other models, use the existing approach
        model = TransferLearningModel(model_name)
        checkpoint = torch.load(f'best_{model_name}_model.pth', map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model

# Image preprocessing function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    return transform(image).unsqueeze(0)

# Grad-CAM functions
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
    elif model_name == 'mobilevit':
        for name, module in model.named_modules():
           if isinstance(module, (nn.Conv2d)):
               return module
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")

def grad_cam(model, input_tensor, model_name, class_idx=None):
    target_layer = get_target_layer(model, model_name)
    gradients = None
    activations = None

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output

    handle1 = target_layer.register_forward_hook(forward_hook)
    handle2 = target_layer.register_backward_hook(backward_hook)

    output = model(input_tensor)
    if class_idx is None:
        class_idx = torch.argmax(output, dim=1).item()
    
    model.zero_grad()
    score = output[:, class_idx]
    score.backward()

    with torch.no_grad():
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1).squeeze(0)
        cam = torch.relu(cam)
        cam = cam.cpu().numpy()

    handle1.remove()
    handle2.remove()

    cam = cv2.resize(cam, (input_tensor.size(3), input_tensor.size(2)))
    cam = (cam - cam.min()) / (cam.max() - cam.min())
    
    return cam

def visualize_grad_cam(orig_image, cam, figsize=(15, 5)):
    plt.figure(figsize=figsize)
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(orig_image, cmap='gray')
    plt.title('Original Image', fontsize=12)
    plt.axis('off')
    
    # Heatmap
    plt.subplot(1, 3, 2)
    plt.imshow(cam, cmap='jet')
    plt.title('Grad-CAM Heatmap', fontsize=12)
    plt.axis('off')
    
    # Overlay
    plt.subplot(1, 3, 3)
    plt.imshow(orig_image, cmap='gray')
    plt.imshow(cam, cmap='jet', alpha=0.5)
    plt.title('Overlay', fontsize=12)
    plt.axis('off')
    
    plt.tight_layout()
    return plt

# Main Streamlit app
def main():
    st.title("OCT Image Classification")

    # Model selection
    model_name = st.selectbox(
        "Select a model for classification", 
        ['mobilenet_v3', 'efficientnet_b0', 'squeezenet1_1', 'resnet50', 'resnet18', 'mobilevit']
    )
    model = load_model(model_name)

    # Image upload
    uploaded_file = st.file_uploader("Upload an OCT image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_container_width=True)

        # Preprocess the image
        input_tensor = preprocess_image(image).to(device)

        # Perform inference
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted].item() * 100

        # Define class labels
        classes = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

        # Display result
        st.write(f"**Prediction:** {classes[predicted]} ({confidence:.2f}% confidence)")

        # Generate Grad-CAM
        cam = grad_cam(model, input_tensor, model_name)
        
        # Display Grad-CAM results
        st.markdown("---")
        st.subheader("Grad-CAM Visualization")
        
        # Convert tensor to numpy for visualization
        img_np = input_tensor.squeeze().cpu().numpy()
        
        # Create and display visualization with larger size
        fig = visualize_grad_cam(img_np, cam, figsize=(15, 5))
        st.pyplot(fig)

        # Additional visualization options
        st.markdown("### Visualization Controls")
        overlay_opacity = st.slider("Overlay Opacity", 0.0, 1.0, 0.5, 0.1)
        
        # Create enhanced overlay with user-controlled opacity
        fig_enhanced = plt.figure(figsize=(12, 4))
        plt.imshow(img_np, cmap='gray')
        plt.imshow(cam, cmap='jet', alpha=overlay_opacity)
        plt.axis('off')
        plt.title('Enhanced Overlay (Adjustable Opacity)')
        st.pyplot(fig_enhanced)

        # Display test accuracy
        if model_name == 'mobilevit':
            metrics_path = 'best_mobilevit_s_model.pth'
        else:
            metrics_path = f'best_{model_name}_model.pth'
            
        metrics = torch.load(metrics_path, map_location=device)
        if 'accuracy' in metrics:
            st.write(f"**Test Accuracy:** {metrics['accuracy']:.2f}%")
        else:
            st.write("**Test Accuracy:** Not available")

if __name__ == '__main__':
    main()