import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="YOLOv8 Object Detection",
    page_icon="🎯",
    layout="wide"
)

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    model = YOLO("models/best.pt")
    return model

model = load_model()

# -------------------------------
# Title
# -------------------------------
st.title("🎯 YOLOv8 Object Detection App")

st.markdown("""
Upload an image and detect objects using your trained **YOLOv8 model**.
""")

# -------------------------------
# Upload Image
# -------------------------------
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------
# Prediction
# -------------------------------
if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file)

    # Convert to RGB
    image = image.convert("RGB")

    # Convert to numpy
    img = np.array(image)

    # Display uploaded image
    st.subheader("Uploaded Image")

    st.image(
        image,
        use_container_width=True
    )

    # Button
    if st.button("Detect Objects"):

        with st.spinner("Detecting objects..."):

            # Prediction
            results = model(img)

            # Draw bounding boxes
            result_img = results[0].plot()

            st.subheader("Detection Result")

            st.image(
                result_img,
                use_container_width=True
            )

            # Display detected objects
            st.subheader("Detected Objects")

            boxes = results[0].boxes

            if len(boxes) > 0:

                for box in boxes:

                    cls_id = int(box.cls[0])

                    confidence = float(box.conf[0])

                    class_name = model.names[cls_id]

                    st.write(
                        f"✅ **{class_name}** : {confidence:.2f}"
                    )

            else:

                st.warning("No objects detected.")

            st.success("Detection Completed Successfully!")