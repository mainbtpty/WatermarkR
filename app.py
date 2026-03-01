# Requirements:
# pip install streamlit opencv-python pytesseract pillow streamlit-drawable-canvas
# Also, install Tesseract OCR on your system and set the path if needed (pytesseract.pytesseract.tesseract_cmd = '/path/to/tesseract')

# Requirements (add to your requirements.txt):
# streamlit==1.40.0
# opencv-python-headless
# pytesseract
# pillow
# streamlit-drawable-canvas
# moviepy  # NEW: For handling video with audio

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pytesseract
from streamlit_drawable_canvas import st_canvas
import tempfile
import os
from moviepy.editor import VideoFileClip  # NEW: For audio handling

st.title("Video Watermark Remover App (with Clone Tool)")

st.write("""
Upload a video to remove watermarks. Supports automatic OCR detection and manual tools.
Now includes a Clone Tool: Select a clean source area, copy it, and paste over the watermark (helpful for moving watermarks by editing per frame).
Watermarks can be static or moving—use frame navigation for per-frame edits.
Output video now preserves original audio.
For advanced AI inpainting, extend with models like LaMa.
""")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])

if uploaded_file is not None:
    # Save uploaded file to temporary path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        video_path = tmp_file.name

    # Load video with MoviePy for audio and metadata (NEW)
    clip = VideoFileClip(video_path)
    fps = clip.fps
    duration = clip.duration
    audio = clip.audio  # Preserve audio

    # OpenCV capture for frames
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Preview original video
    st.subheader("Original Video Preview")
    st.video(video_path)

    # Frame navigation (NEW: For per-frame editing, especially for moving watermarks)
    st.subheader("Frame Selection")
    selected_frame = st.slider("Select Frame for Editing", 1, frame_count, 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, selected_frame - 1)
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to read frame.")
        cap.release()
        os.unlink(video_path)
        st.stop()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    # Automatic watermark detection using Tesseract OCR
    st.subheader("Automatic Watermark Detection (OCR)")
    if st.button("Detect Text Watermarks"):
        with st.spinner("Detecting watermarks..."):
            data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
            detected = []
            for i in range(len(data['level'])):
                conf = int(data['conf'][i])
                if conf > 60:
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    text = data['text'][i].strip()
                    if text:
                        detected.append({"text": text, "box": (x, y, w, h)})
                        st.write(f"Detected: '{text}' at (x={x}, y={y}, w={w}, h={h}) with conf {conf}")

            if detected:
                st.success(f"Found {len(detected)} potential watermarks.")
            else:
                st.info("No text watermarks detected.")

    # Manual tools section
    st.subheader("Manual Watermark Removal Tools")
    st.write("""
    Use tools to highlight or clone areas on the selected frame.
    - Inpainting Mask: Draw masks for AI inpainting (erases and fills).
    - Clone Tool: Copy a clean area and paste over watermark (for moving logos, edit per frame).
    Changes apply to the current frame; for video-wide, process after editing key frames.
    """)

    # Session state for per-frame masks and clones (NEW: Store edits per frame)
    if 'frame_edits' not in st.session_state:
        st.session_state.frame_edits = {f: {'mask': None, 'clones': []} for f in range(1, frame_count + 1)}

    # Tool selection
    tool_mode = st.selectbox("Select Tool Mode", ["Inpainting Mask", "Clone Tool"])

    if tool_mode == "Inpainting Mask":
        drawing_mode = st.selectbox("Drawing Tool", ["rect", "freedraw", "transform"])
        stroke_width = st.slider("Stroke/Brush Width", 1, 50, 10)
        stroke_color = st.color_picker("Stroke Color", "#FF0000")

        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_image=pil_image,
            update_streamlit=True,
            height=height,
            width=width,
            drawing_mode=drawing_mode,
            key=f"mask_canvas_{selected_frame}",
        )

        if canvas_result.image_data is not None:
            canvas_image = np.array(canvas_result.image_data)
            mask = np.any(canvas_image[:, :, :3] != [255, 255, 255], axis=-1).astype(np.uint8) * 255
            st.session_state.frame_edits[selected_frame]['mask'] = mask

    elif tool_mode == "Clone Tool":
        st.write("1. Draw a rectangle to select SOURCE area (clean pixels to copy).")
        st.write("2. Then draw another for TARGET area (paste over watermark).")
        st.write("Click 'Apply Clone' to preview on this frame.")

        clone_drawing_mode = "rect"  # Fixed to rect for simplicity
        stroke_width = 1
        stroke_color = "#00FF00"  # Green for clone

        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_image=pil_image,
            update_streamlit=True,
            height=height,
            width=width,
            drawing_mode=clone_drawing_mode,
            key=f"clone_canvas_{selected_frame}",
        )

        # Extract rectangles from canvas (assuming user draws two rects: source then target)
        if canvas_result.json_data is not None and 'objects' in canvas_result.json_data:
            objects = canvas_result.json_data['objects']
            rects = [obj for obj in objects if obj['type'] == 'rect']
            if len(rects) >= 2:
                source_rect = rects[-2]  # Second last: source
                target_rect = rects[-1]  # Last: target
                src_x, src_y = int(source_rect['left']), int(source_rect['top'])
                src_w, src_h = int(source_rect['width']), int(source_rect['height'])
                tgt_x, tgt_y = int(target_rect['left']), int(target_rect['top'])
                tgt_w, tgt_h = int(target_rect['width']), int(target_rect['height'])

                if st.button("Apply Clone to This Frame"):
                    # Copy source to target with blending (NEW: Clone logic)
                    source_patch = frame[src_y:src_y+src_h, src_x:src_x+src_w]
                    if source_patch.shape[:2] != (tgt_h, tgt_w):
                        source_patch = cv2.resize(source_patch, (tgt_w, tgt_h))
                    frame[tgt_y:tgt_y+tgt_h, tgt_x:tgt_x+tgt_w] = source_patch
                    # Optional: Blend edges for perfection (using seamlessClone)
                    center = (tgt_x + tgt_w // 2, tgt_y + tgt_h // 2)
                    cloned = cv2.seamlessClone(source_patch, frame, np.ones_like(source_patch) * 255, center, cv2.NORMAL_CLONE)
                    frame = cloned

                    # Save clone op (but for now, apply directly; for multi, store)
                    st.session_state.frame_edits[selected_frame]['clones'].append({
                        'src': (src_x, src_y, src_w, src_h),
                        'tgt': (tgt_x, tgt_y, tgt_w, tgt_h)
                    })

                    # Show preview of cloned frame
                    cloned_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    st.image(cloned_rgb, caption="Cloned Frame Preview")

    # Process entire video
    if st.button("Process Video (Apply Edits to All Frames)"):
        output_path_no_audio = os.path.join(tempfile.gettempdir(), "processed_no_audio.mp4")
        output_path = os.path.join(tempfile.gettempdir(), "processed_video.mp4")
        out = cv2.VideoWriter(output_path_no_audio, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        progress_bar = st.progress(0)

        with st.spinner("Processing frames..."):
            for i in range(1, frame_count + 1):
                ret, frame = cap.read()
                if not ret:
                    break

                edits = st.session_state.frame_edits.get(i, {'mask': None, 'clones': []})

                # Apply mask inpainting if exists
                if edits['mask'] is not None:
                    frame = cv2.inpaint(frame, edits['mask'], 3, cv2.INPAINT_NS)

                # Apply clones if any (NEW)
                for clone in edits['clones']:
                    src_x, src_y, src_w, src_h = clone['src']
                    tgt_x, tgt_y, tgt_w, tgt_h = clone['tgt']
                    source_patch = frame[src_y:src_y+src_h, src_x:src_x+src_w]  # Note: source from original frame?
                    if source_patch.shape[:2] != (tgt_h, tgt_w):
                        source_patch = cv2.resize(source_patch, (tgt_w, tgt_h))
                    center = (tgt_x + tgt_w // 2, tgt_y + tgt_h // 2)
                    frame = cv2.seamlessClone(source_patch, frame, np.ones_like(source_patch) * 255, center, cv2.NORMAL_CLONE)

                out.write(frame)
                progress_bar.progress(i / frame_count)

        out.release()
        cap.release()

        # Add audio back (NEW: Using MoviePy)
        processed_clip = VideoFileClip(output_path_no_audio)
        processed_clip = processed_clip.set_audio(audio)
        processed_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')

        # Preview and download
        st.subheader("Processed Video Preview")
        st.video(output_path)

        with open(output_path, "rb") as f:
            st.download_button("Download Processed Video", f, "removed_watermark.mp4", "video/mp4")

        # Cleanup
        os.unlink(video_path)
        os.unlink(output_path_no_audio)
        os.unlink(output_path)

else:
    st.info("Upload a video to start.")
