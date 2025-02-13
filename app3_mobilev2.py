import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIL import Image

# Load the pretrained MobileNetV2 model
@st.cache_resource
def load_model():
    num_classes = 8  # Replace with your dataset's number of classes
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)  # Modify the classifier
    model.load_state_dict(torch.load("transfer_learning_mobilenet_v2.pth", map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    model.eval()
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model

# Define the image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Main app
st.set_page_config(page_title="Cucumber Leaf Disease Detection", layout="wide")

# Title and subtitle (fixed at the top of the page)
st.markdown(
    """
    <div style='text-align: center; font-size: 32px; font-weight: bold;'>Explainable AI for Interpretable Leaf Disease Detection</div>
    <div style='text-align: center; font-size: 20px; font-weight: normal; margin-top: -10px;'>Usecase: Cucumber Disease Detection</div>
    """,
    unsafe_allow_html=True,
)

# Centered instructions
st.markdown(
    """
    <div style='text-align: center; font-size: 18px; margin-top: 20px;'>
        Upload an image to classify and visualize <b>Grad-CAM</b>, <b>Grad-CAM++</b>, and <b>Eigen-CAM</b>.
    </div>
    """,
    unsafe_allow_html=True,
)

# Right sidebar for the upload option
st.sidebar.header("Upload Your Image")
uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    original_image_np = np.array(image).astype(np.float32) / 255.0

    # Transform image
    transformed_image = transform(image).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model()

    # Perform inference
    with torch.no_grad():
        outputs = model(transformed_image)
        predicted_class = outputs.argmax().item()

    class_names = ["Anthracnose", "Bacterial Wilt", "Belly Rot", "Downy Mildew", "Fresh Cucumber", "Fresh Leaf", "Gummy Stem Blight", "Pythium Fruit Rot"]

    # Display the prediction in a bordered box
    st.sidebar.markdown(
        f"""
        <div style='border: 2px solid #4CAF50; border-radius: 10px; padding: 15px; text-align: center; background-color: #f9f9f9;'>
            <h3 style='color: #333;'>Prediction</h3>
            <p style='font-size: 18px; font-weight: bold; color: #4CAF50;'>{class_names[predicted_class]}</p>
            <p style='font-size: 14px; color: #555;'>(Class ID: {predicted_class})</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Grad-CAM, Grad-CAM++, Eigen-CAM
    target_layers = [model.features[-1]]  # MobileNetV2 uses 'features' for the last convolutional layer

    # Generate CAM visualizations
    target = [ClassifierOutputTarget(predicted_class)]

    # Original Image
    grid_images = [np.array(image)]
    captions = [f"**Original Image**\n(Predicted: {class_names[predicted_class]})"]

    # Grad-CAM
    gradcam = GradCAM(model=model, target_layers=target_layers)
    gradcam_heatmap = gradcam(input_tensor=transformed_image, targets=target)[0]
    gradcam_result = show_cam_on_image(original_image_np, cv2.resize(gradcam_heatmap, (original_image_np.shape[1], original_image_np.shape[0])), use_rgb=True)
    grid_images.append(gradcam_result)
    captions.append(f"**Grad-CAM**\n(Predicted: {class_names[predicted_class]})")

    # Grad-CAM++
    gradcam_plus_plus = GradCAMPlusPlus(model=model, target_layers=target_layers)
    gradcam_plus_plus_heatmap = gradcam_plus_plus(input_tensor=transformed_image, targets=target)[0]
    gradcam_plus_plus_result = show_cam_on_image(original_image_np, cv2.resize(gradcam_plus_plus_heatmap, (original_image_np.shape[1], original_image_np.shape[0])), use_rgb=True)
    grid_images.append(gradcam_plus_plus_result)
    captions.append(f"**Grad-CAM++**\n(Predicted: {class_names[predicted_class]})")

    # Eigen-CAM
    eigen_cam = EigenCAM(model=model, target_layers=target_layers)
    eigen_cam_heatmap = eigen_cam(input_tensor=transformed_image, targets=target)[0]
    eigen_cam_result = show_cam_on_image(original_image_np, cv2.resize(eigen_cam_heatmap, (original_image_np.shape[1], original_image_np.shape[0])), use_rgb=True)
    grid_images.append(eigen_cam_result)
    captions.append(f"**Eigen-CAM**\n(Predicted: {class_names[predicted_class]})")

    # Centered visualization results title
    st.markdown(
        """
        <div style='text-align: center; font-size: 20px; font-weight: bold; margin-top: 30px;'>
            Visualization Results
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Display results in a 1x4 grid (one row for all images)
    cols = st.columns(4, gap="medium")
    for i, col in enumerate(cols):
        with col:
            # Resize each image to 400x400 and center-align
            st.image(cv2.resize(grid_images[i], (400, 400)), use_column_width=False)
            st.markdown(f"<div style='text-align: center; font-size: 18px; font-weight: bold;'>{captions[i]}</div>", unsafe_allow_html=True)
else:
    st.info("Please upload an image to proceed.")
