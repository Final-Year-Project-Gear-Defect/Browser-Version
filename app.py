import streamlit as st
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from ultralytics import YOLO 

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # make sure best.pt is in same folder

model = load_model()

# Class colors
class_colors = {"kp": (0, 255, 255), "hp_cm": (255, 0, 255), "hp_cd": (128, 0, 255)}

def predict_and_overlay(img):
    result = model.predict(source=img, imgsz=640, conf=0.25, save=False)[0]
    image = result.orig_img.copy()
    
    # Track defect statistics
    defect_counts = {"kp": 0, "hp_cm": 0, "hp_cd": 0}
    defect_confidences = {"kp": [], "hp_cm": [], "hp_cd": []}
    
    for box in result.obb:
        cls_id = int(box.cls[0])
        label = result.names[cls_id]
        confidence = float(box.conf[0])
        pts = box.xyxyxyxy[0].cpu().numpy().astype(int).reshape(-1, 2)

        # Count defects and store confidences
        defect_counts[label] += 1
        defect_confidences[label].append(confidence)

        overlay = image.copy()
        cv2.fillPoly(overlay, [pts], class_colors[label])
        image = cv2.addWeighted(overlay, 0.4, image, 0.6, 0)
        cv2.polylines(image, [pts], isClosed=True, color=class_colors[label], thickness=2)
        x, y = pts[0]
        cv2.putText(image, f"{label} ({confidence:.2f})", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, class_colors[label], 2)

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), defect_counts, defect_confidences


def create_defect_bar_chart(defect_counts):
    """Create a bar chart showing defect counts"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define defect labels with better names
    defect_labels = {
        "kp": "Key Point Defects",
        "hp_cm": "High Precision Common",
        "hp_cd": "High Precision Critical"
    }
    
    labels = [defect_labels[key] for key in defect_counts.keys()]
    counts = list(defect_counts.values())
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    bars = ax.bar(labels, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax.set_ylabel('Number of Defects', fontsize=12, fontweight='bold')
    ax.set_title('Gear Defect Analysis - Count by Type', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on top of bars
    for bar, count in zip(bars, counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                   str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig


def create_defect_pie_chart(defect_counts):
    """Create a pie chart showing defect distribution"""
    # Filter out zero counts
    filtered_counts = {k: v for k, v in defect_counts.items() if v > 0}
    
    if not filtered_counts:
        return None
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    defect_labels = {
        "kp": "Key Point Defects",
        "hp_cm": "High Precision Common",
        "hp_cd": "High Precision Critical"
    }
    
    labels = [defect_labels[key] for key in filtered_counts.keys()]
    sizes = list(filtered_counts.values())
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1'][:len(labels)]
    
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                      startangle=90, textprops={'fontsize': 10})
    
    # Beautify the percentage text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax.set_title('Defect Distribution', fontsize=14, fontweight='bold', pad=20)
    return fig


def create_confidence_analysis(defect_confidences):
    """Create confidence level analysis chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    defect_labels = {
        "kp": "Key Point Defects",
        "hp_cm": "High Precision Common", 
        "hp_cd": "High Precision Critical"
    }
    
    # Calculate average confidence for each defect type
    avg_confidences = {}
    for defect_type, confidences in defect_confidences.items():
        if confidences:
            avg_confidences[defect_labels[defect_type]] = np.mean(confidences)
    
    if avg_confidences:
        labels = list(avg_confidences.keys())
        confidences = list(avg_confidences.values())
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1'][:len(labels)]
        
        bars = ax.bar(labels, confidences, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax.set_ylabel('Average Confidence Score', fontsize=12, fontweight='bold')
        ax.set_title('Detection Confidence Analysis', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels
        for bar, conf in zip(bars, confidences):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{conf:.3f}', ha='center', va='bottom', fontweight='bold')
    
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        return fig
    
    return None


# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="⚙️ Gear Fault Detection", page_icon="🛠", layout="centered")
st.title("⚙️ Gear Fault Detection Software")

uploaded_file = st.file_uploader("📂 Upload Gear Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Convert uploaded file to image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # Show uploaded image
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="📸 Uploaded Image", use_column_width=True)

    if st.button("🔍 Run Detection"):
        with st.spinner("🔄 Detecting faults... Please wait"):
            time.sleep(1)
            result_img, defect_counts, defect_confidences = predict_and_overlay(img)

        st.success("✅ Detection Completed!")
        
        # Display results in columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(result_img, caption="🛠 Detection Result", use_column_width=True)
        
        with col2:
            total_defects = sum(defect_counts.values())
            st.metric("🔍 Total Defects Found", total_defects)
            
            # Show defect breakdown
            st.subheader("📊 Defect Breakdown")
            defect_labels = {
                "kp": "🔴 Key Point Defects",
                "hp_cm": "🟡 High Precision Common",
                "hp_cd": "🔵 High Precision Critical"
            }
            
            for defect_type, count in defect_counts.items():
                if count > 0:
                    st.write(f"{defect_labels[defect_type]}: **{count}**")

        # Visualization Section
        if total_defects > 0:
            st.header("📈 Defect Analysis Visualizations")
            
            # Create tabs for different visualizations
            tab1, tab2, tab3 = st.tabs(["📊 Bar Chart", "🥧 Distribution", "🎯 Confidence Analysis"])
            
            with tab1:
                st.subheader("Defect Count Analysis")
                fig_bar = create_defect_bar_chart(defect_counts)
                st.pyplot(fig_bar)
                plt.close(fig_bar)  # Close to free memory
            
            with tab2:
                st.subheader("Defect Distribution")
                fig_pie = create_defect_pie_chart(defect_counts)
                if fig_pie:
                    st.pyplot(fig_pie)
                    plt.close(fig_pie)  # Close to free memory
                else:
                    st.info("No defects detected for pie chart visualization.")
            
            with tab3:
                st.subheader("Detection Confidence Analysis")
                fig_conf = create_confidence_analysis(defect_confidences)
                if fig_conf:
                    st.pyplot(fig_conf)
                    plt.close(fig_conf)  # Close to free memory
                else:
                    st.info("No confidence data available.")
        # Save + Download button
        save_path = "detection_result.jpg"
        cv2.imwrite(save_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
        with open(save_path, "rb") as f:
            st.download_button("⬇️ Download Result", f, file_name="gear_fault_result.jpg", mime="image/jpeg")
