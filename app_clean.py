import streamlit as st
import cv2
import numpy as np
import time
from ultralytics import YOLO 
from datetime import datetime

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# Class colors
class_colors = {"kp": (0, 255, 255), "hp_cm": (255, 0, 255), "hp_cd": (128, 0, 255)}

# Simple CSS
def load_css():
    st.markdown("""
    <style>
    .main > div {
        padding-top: 2rem;
        max-width: 800px;
        margin: 0 auto;
    }
    
    .title {
        text-align: center;
        color: #2c3e50;
        margin-bottom: 2rem;
    }
    
    .result-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        margin: 1rem 0;
    }
    
    .stButton > button {
        width: 100%;
        background-color: #007bff;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

def predict_and_overlay(img):
    result = model.predict(source=img, imgsz=640, conf=0.25, save=False)[0]
    image = result.orig_img.copy()
    fault_count = 0

    for box in result.obb:
        cls_id = int(box.cls[0])
        label = result.names[cls_id]
        confidence = float(box.conf[0])
        pts = box.xyxyxyxy[0].cpu().numpy().astype(int).reshape(-1, 2)
        fault_count += 1

        overlay = image.copy()
        cv2.fillPoly(overlay, [pts], class_colors[label])
        image = cv2.addWeighted(overlay, 0.4, image, 0.6, 0)
        cv2.polylines(image, [pts], isClosed=True, color=class_colors[label], thickness=2)
        
        x, y = pts[0]
        cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, class_colors[label], 2)

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), fault_count

# Main UI
st.set_page_config(
    page_title="Gear Fault Detection",
    page_icon="⚙️",
    layout="centered"
)

load_css()

# Header
st.markdown("""
<div class="title">
    <h1>⚙️ Gear Fault Detection</h1>
    <p>Upload a gear image to detect faults</p>
</div>
""", unsafe_allow_html=True)

# File upload
uploaded_file = st.file_uploader(
    "Choose an image file",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Convert to image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # Display original image
    st.subheader("📸 Original Image")
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_column_width=True)
    
    # Run detection button
    if st.button("🔍 Detect Faults"):
        with st.spinner("Processing..."):
            time.sleep(1)
            result_img, fault_count = predict_and_overlay(img)
        
        # Show results
        st.subheader("🔧 Detection Results")
        st.image(result_img, use_column_width=True)
        
        # Summary
        st.markdown(f"""
        <div class="result-box">
            <h4>Summary:</h4>
            <p><strong>Faults detected:</strong> {fault_count}</p>
            <p><strong>Status:</strong> {'❌ Faults found' if fault_count > 0 else '✅ No faults detected'}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Download button
        save_path = f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(save_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
        
        with open(save_path, "rb") as f:
            st.download_button(
                "📥 Download Result",
                f,
                file_name=save_path,
                mime="image/jpeg"
            )

# Footer
st.markdown("---")
st.markdown("*Gear Fault Detection System - Simple & Clean Interface*")