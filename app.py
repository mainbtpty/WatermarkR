# Requirements:
# pip install streamlit opencv-python pytesseract pillow streamlit-drawable-canvas
# Also, install Tesseract OCR on your system and set the path if needed (pytesseract.pytesseract.tesseract_cmd = '/path/to/tesseract')

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pytesseract
from streamlit_drawable_canvas import st_canvas
import tempfile
import os

st.title("Video Watermark Remover App")

st.write("""
Upload a video to remove watermarks. The app supports automatic detection using OCR (Tesseract) and manual tools like rectangle, eraser/brush for highlighting areas.
Watermarks are assumed to be static across frames. Processing uses OpenCV's inpainting for removal.
For better AI-based inpainting, you can extend this with models like LaMa (requires additional setup).
""")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])

if uploaded_file is not None:
    # Save uploaded file to temporary path for OpenCV processing
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        video_path = tmp_file.name

    # Preview original video
    st.subheader("Original Video Preview")
    st.video(video_path)

    # Load video capture
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to read the video file.")
        cap.release()
        os.unlink(video_path)
        st.stop()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    height, width = frame.shape[:2]

    # Automatic watermark detection using Tesseract OCR
    st.subheader("Automatic Watermark Detection (OCR)")
    if st.button("Detect Text Watermarks"):
        with st.spinner("Detecting watermarks..."):
            # Use Tesseract to detect text in the frame
            data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
            detected = []
            for i in range(len(data['level'])):
                conf = int(data['conf'][i])
                if conf > 60:  # Confidence threshold
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    text = data['text'][i].strip()
                    if text:  # Ignore empty text
                        detected.append({"text": text, "box": (x, y, w, h)})
                        st.write(f"Detected: '{text}' at position (x={x}, y={y}, w={w}, h={h}) with confidence {conf}")

            if detected:
                st.success(f"Found {len(detected)} potential text watermarks.")
                # For simplicity, we could auto-generate a mask from all detected boxes, but here we list them.
                # To auto-remove, user can proceed to manual and draw over them.
            else:
                st.info("No text watermarks detected with high confidence.")

    # Manual watermark highlighting using drawable canvas
    st.subheader("Manual Watermark Removal Tools")
    st.write("""
    Use the tools below to highlight watermark areas on a sample frame.
    - Rectangle: Draw adjustable rectangles to cover areas.
    - Freedraw: Use as brush/eraser to freely draw over watermarks (similar to unwatermark.ai's brush).
    - Transform: Adjust/resize drawn shapes.
    """)

    drawing_mode = st.selectbox("Select Tool", ["rect", "freedraw", "transform"])
    stroke_width = st.slider("Stroke/Brush Width", 1, 50, 10)
    stroke_color = st.color_picker("Stroke Color", "#FF0000")

    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",  # Transparent fill
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_image=pil_image,
        update_streamlit=True,
        height=height,
        width=width,
        drawing_mode=drawing_mode,
        key="watermark_canvas",
    )

    # Process video if mask is drawn and button pressed
    if canvas_result.image_data is not None and st.button("Remove Watermark and Process Video"):
        # Create mask from canvas (where alpha > 0 or stroke is drawn)
        canvas_image = np.array(canvas_result.image_data)
        # Mask is where there's drawing (non-background)
        mask = np.any(canvas_image[:, :, :3] != [255, 255, 255], axis=-1).astype(np.uint8) * 255  # Assuming white background

        if np.sum(mask) == 0:
            st.warning("No areas highlighted. Please draw over the watermark.")
        else:
            # Prepare output video
            output_path = os.path.join(tempfile.gettempdir(), "processed_video.mp4")
            fps = cap.get(cv2.CAP_PROP_FPS)
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

            # Reset capture to start
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            progress_bar = st.progress(0)
            st.write(f"Processing {frame_count} frames...")

            with st.spinner("Removing watermarks..."):
                for i in range(frame_count):
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Apply inpainting (AI-assisted via OpenCV's method; can replace with torch-based model)
                    inpainted_frame = cv2.inpaint(frame, mask, 3, cv2.INPAINT_NS)  # Or INPAINT_TELEA for alternative method

                    out.write(inpainted_frame)
                    progress_bar.progress((i + 1) / frame_count)

            out.release()
            cap.release()

            # Preview and download processed video
            st.subheader("Processed Video Preview")
            st.video(output_path)

            with open(output_path, "rb") as f:
                st.download_button(
                    label="Download Processed Video",
                    data=f,
                    file_name="removed_watermark.mp4",
                    mime="video/mp4"
                )

            # Cleanup temp files
            os.unlink(video_path)
            os.unlink(output_path)

else:
    st.info("Please upload a video file to get started.")
