import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

# Title of the application
st.title("Car Number Plate Detection")

# Path to Haar Cascade XML file
CASCADE_PATH = os.path.join("model", "haarcascade_russian_plate_number.xml")

# Check if Haar Cascade file exists
if not os.path.exists(CASCADE_PATH):
    st.error("Haar Cascade XML file not found! Ensure the file is in the 'model' folder.")
else:
    plate_cascade = cv2.CascadeClassifier(CASCADE_PATH)

    # Function to detect number plates
    def detect_number_plate(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        plates = plate_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
        )
        for (x, y, w, h) in plates:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return image, len(plates)

    # Upload image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        try:
            # Read the uploaded image
            image = Image.open(uploaded_file)
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV

            # Detect plates
            processed_image, plate_count = detect_number_plate(image)

            # Display results
            st.subheader(f"Detected Plates: {plate_count}")
            st.image(processed_image, channels="BGR", caption="Processed Image")
        except Exception as e:
            st.error(f"Error processing the image: {e}")

    # Live Video Detection
    st.header("Live Video Detection (Optional)")
    run_live_detection = st.checkbox("Enable Webcam Detection")

    if run_live_detection:
        st.warning("This feature requires local execution and a working webcam.")
        st.write("Press 'Stop' to end the video stream.")

        # Open webcam stream
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not access webcam. Ensure it's connected and enabled.")
        else:
            frame_placeholder = st.empty()  # Placeholder for displaying video frames
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to capture video frame.")
                        break

                    # Process each frame
                    processed_frame, _ = detect_number_plate(frame)

                    # Display frame in Streamlit
                    frame_placeholder.image(processed_frame, channels="BGR")

                    # Check for stop button to exit the loop
                    if not run_live_detection:
                        break

            except Exception as e:
                st.error(f"Error during live detection: {e}")
            finally:
                cap.release()
