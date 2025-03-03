# Deep Learning-Driven Detection and Classification of Retinal Conditions from OCT Images

## 📌 Project Overview
This project leverages **Deep Learning (DL)** and **Explainable AI (XAI)** for the detection and classification of retinal diseases using **Optical Coherence Tomography (OCT) images**. The model is trained to classify four categories of retinal conditions:  
- **NORMAL (Healthy Retina)**
- **Choroidal Neovascularization (CNV)**
- **Diabetic Macular Edema (DME)**
- **Drusen**

By utilizing advanced **transfer learning models**, the project aims to improve the precision, efficiency, and interpretability of **OCT-based disease classification**, assisting ophthalmologists in accurate and early diagnosis.  

## 🔍 Key Features
✅ **Transfer Learning Models**: Implemented **ResNet18, ResNet50, EfficientNet-B0, SqueezeNet1.1, MobileNetV3, and MobileViT-S** to classify OCT images.  
✅ **High Classification Accuracy**: **EfficientNet-B0 (96.49%)** and **ResNet18 (95.45%)** achieved the highest classification scores.  
✅ **Explainable AI (XAI)**: Integrated **Grad-CAM** visualization to provide heatmaps, making AI predictions more interpretable for clinicians.  
✅ **Optimized for Real-Time Use**: Lightweight architectures like **SqueezeNet1.1 (91.49%)** and **MobileNetV3 (92.98%)** ensure faster inference and deployment feasibility.  
✅ **Benchmarking Performance**: Evaluated models based on **accuracy, precision, recall, F1-score, and confusion matrices** to compare their effectiveness.  

## 📂 Dataset Used
- **Dataset**: OCT2017 dataset  
- **Classes**: NORMAL, CNV, DME, DRUSEN  
- **Preprocessing Steps**: Image resizing, augmentation, normalization  

## 🚀 Technologies Used
- **Deep Learning Frameworks**: TensorFlow, PyTorch  
- **Model Architectures**: CNN-based Transfer Learning Models  
- **Explainability Tools**: Grad-CAM  
- **Libraries**: OpenCV, NumPy, Matplotlib, Pandas, Scikit-learn  

## 💻 Installation & Setup
1. **Clone the Repository**  
   ```bash
   git clone https://github.com/indraneel03.git
   cd eye

 
 📊 Model Performance
Model	Accuracy (%)
EfficientNet-B0	96.49
ResNet18	95.45
MobileViT-S	97.83
ResNet50	93.08
MobileNetV3	92.98
SqueezeNet1.1	91.49

📜 Contributors
Indraneel Kalva (Anurag University)
Mothe Manoj Reddy (Anurag University)
Varkala Pranavi Goud (Anurag University)
Mr. Rambabu Atmakuri (Assistant Professor, Anurag University)

🏆 Future Enhancements
Expand dataset for better generalization.
Implement attention mechanisms for improved feature extraction.
Deploy the model as a web application for real-time diagnosis.
📬 Contact
📧 indraneelkalva@gmail.com 

