import streamlit as st
import cv2
import numpy as np
import tempfile
from PIL import Image
import os
import io
from pathlib import Path

# Import từ các file module đã chia
from haar_detector import load_haar, detect_faces_haar
from dnn_detector import load_dnn, detect_faces_dnn
from apply_blur import apply_blur_to_image

st.set_page_config(page_title="Face Blurring (Privacy Filter)", layout="centered")

# ---------------- UI ----------------

st.title("🔒 Face Blurring — Privacy Filter")
st.write("Upload an image or video. Detect faces (Haar / DNN) and Blur (Gaussian or Pixelate).")

with st.sidebar:
    st.header("Settings")

    detector = st.selectbox("Face detector", ("Haar Cascade (fast)", "DNN (more accurate)"))
    blur_method = st.selectbox("Blur method", ("Gaussian", "Pixelate"))
    draw_boxes = st.checkbox("Draw detection boxes (for debug)", value=False)

    gaussian_kernel = (51, 51)
    pixel_blocks = 10

    if blur_method == "Gaussian":
        k = st.slider("Gaussian kernel size (odd)", 11, 101, 51, step=2)
        gaussian_kernel = (k, k)
    else:
        pixel

    st.markdown("---")
    st.write("DNN model (optional):")
    prototxt_path = st.text_input("Prototxt path (deploy.prototxt)", "")
    model_path = st.text_input("Caffe model path (res10_300x300.caffemodel)", "")
    st.info("Nếu chọn DNN thì nhập đủ 2 file model.")

tab1, tab2, tab3 = st.tabs(["Upload File", "Chụp từ Webcam", "Webcam Live Blur (Realtime)"])

# Biến uploaded vẫn giữ nguyên để code cũ chạy ngon
uploaded = None

# ==================== TAB 1: UPLOAD FILE  ====================
with tab1:
    uploaded = st.file_uploader(
        "Choose an image or video",
        type=["jpg","jpeg","png","bmp","mp4","mov","avi","mkv"],
        accept_multiple_files=False
    )

# ==================== TAB 2: CHỤP ẢNH TỪ WEBCAM  ====================
with tab2:
    st.write("### 2) Chụp ảnh từ Webcam")
    camera_img = st.camera_input("Nhấn để chụp", key="static_cam")
    if camera_img:
        uploaded = camera_img  # gán vào uploaded → code xử lý cũ vẫn chạy bình thường
        st.success("Đã chụp từ webcam!")

# ==================== TAB 3: WEBCAM REALTIME BLUR ====================
with tab3:
    st.write("### Webcam Live – Làm mờ khuôn mặt realtime")
    st.info("Camera sẽ bật ngay – khuôn mặt sẽ bị làm mờ trực tiếp!")

    # Khởi tạo model một lần duy nhất
    if "face_detector" not in st.session_state:
        if detector == "DNN (more accurate)" and prototxt_path and model_path:
            try:
                st.session_state.face_detector = load_dnn(prototxt_path, model_path)
                st.session_state.detector_type = "dnn"
                st.success("Đã tải DNN model thành công!")
            except:
                st.error("Lỗi tải DNN model → chuyển về Haar Cascade")
                st.session_state.face_detector = load_haar()
                st.session_state.detector_type = "haar"
        else:
            st.session_state.face_detector = load_haar()
            st.session_state.detector_type = "haar"

    # Frame placeholder để hiển thị realtime
    frame_placeholder = st.empty()

    # Nút bật/tắt webcam realtime
    if st.button("Bật Webcam Realtime", type="primary", use_container_width=True):
        st.session_state.run_webcam = True
    if st.button("Tắt Webcam", use_container_width=True):
        st.session_state.run_webcam = False

    # Chạy webcam realtime
    if st.session_state.get("run_webcam", False):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while st.session_state.run_webcam:
            ret, frame = cap.read()
            if not ret:
                st.error("Không thể truy cập webcam!")
                break

            # Phát hiện khuôn mặt
            if st.session_state.detector_type == "dnn":
                boxes = detect_faces_dnn(frame, st.session_state.face_detector, conf_threshold=0.5)
            else:
                boxes = detect_faces_haar(frame, st.session_state.face_detector)

            # Làm mờ khuôn mặt
            blurred_frame = apply_blur_to_image(
                frame,
                boxes,
                method="gaussian" if blur_method == "Gaussian" else "pixelate",
                gaussian_kernel=gaussian_kernel,
                pixel_blocks=pixel_blocks,
                draw_boxes=draw_boxes
            )

            # Chuyển BGR → RGB để hiển thị đúng màu
            blurred_frame_rgb = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2RGB)

            # Hiển thị frame realtime
            frame_placeholder.image(blurred_frame_rgb, channels="RGB", use_column_width=True)

        cap.release()
        frame_placeholder.empty()  # xóa khung hình khi tắt

# ---------------- Session state ----------------
for key in ["original_img", "processed_img", "original_video_file", "processed_video_file"]:
    if key not in st.session_state:
        st.session_state[key] = None

# Nếu bạn có thêm dòng kiểm tra uploaded ở dưới (ví dụ xử lý ảnh/video), 
# thì giờ nó sẽ nhận cả ảnh từ webcam mà KHÔNG CẦN SỬA GÌ HẾT

# ---------------- Load models ----------------

face_cascade = load_haar()
dnn_net = None

if detector.startswith("DNN") and prototxt_path and model_path:
    try:
        dnn_net = load_dnn(prototxt_path, model_path)
    except Exception as e:
        st.error(f"Failed to load DNN: {e}")
        dnn_net = None

#-------
def process_image_file(file_bytes):

    file_bytes.seek(0)
    file_bytes_bytes = np.asarray(bytearray(file_bytes.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes_bytes, cv2.IMREAD_COLOR)
    if img is None:
        st.error("Không thể đọc file ảnh. Hãy thử định dạng khác.")
        return None
    st.session_state.original_img = img

    if detector.startswith("DNN") and dnn_net is not None:
        boxes = detect_faces_dnn(img, dnn_net, conf_threshold=0.5)
    else:
        boxes = detect_faces_haar(img, face_cascade)

    if blur_method == "Gaussian":
        out = apply_blur_to_image(img, boxes, method="gaussian", gaussian_kernel=gaussian_kernel, draw_boxes=draw_boxes)
    else:
        out = apply_blur_to_image(img, boxes, method="pixelate", pixel_blocks=pixel_blocks, draw_boxes=draw_boxes)
    st.session_state.processed_img = out
    return out, boxes
def process_video_file(temp_input_path, temp_output_path):
    cap = cv2.VideoCapture(temp_input_path)
    if not cap.isOpened():
        st.error("Không mở được video.")
        return None

    # Lấy thông tin video gốc
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Fix lỗi số 1: Đảm bảo kích thước đủ lớn để detect
    if width < 600:  # nếu video quá nhỏ (nhiều điện thoại quay 480p)
        scale = 1280 / width
        width = 1280
        height = int(height * scale)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

    progress_bar = st.progress(0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1000  # tránh chia 0
    processed_frames = 0
    detect_every_n_frames = 3  # Fix lỗi số 2: chỉ detect mỗi 3 frame

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame lên để detect chính xác hơn (rất quan trọng!)
        display_frame = frame.copy()
        if frame.shape[1] < 600:  # nếu nhỏ hơn 600px
            frame = cv2.resize(frame, (1280, int(1280 * frame.shape[0] / frame.shape[1])))

        # Chỉ detect mỗi N frame để tăng tốc và ổn định
        if processed_frames % detect_every_n_frames == 0:
            if detector.startswith("DNN") and dnn_net is not None:
                current_boxes = detect_faces_dnn(frame, dnn_net, conf_threshold=0.5)
            else:
                # Tăng độ nhạy cho Haar
                current_boxes = detect_faces_haar(frame, face_cascade, scaleFactor=1.05, minNeighbors=3)
        # Các frame giữa dùng lại boxes cũ (người không di chuyển nhiều)

        # Áp dụng blur lên frame gốc (không resize)
        if len(current_boxes) > 0:
            # Chuyển boxes về tọa độ frame gốc nếu đã resize
            scale_x = display_frame.shape[1] / frame.shape[1]
            scale_y = display_frame.shape[0] / frame.shape[0]
            scaled_boxes = []
            for (x, y, w, h) in current_boxes:
                x = int(x * scale_x)
                y = int(y * scale_y)
                w = int(w * scale_x)
                h = int(h * scale_y)
                scaled_boxes.append((x, y, w, h))
            
            if blur_method == "Gaussian":
                display_frame = apply_blur_to_image(display_frame, scaled_boxes, 
                                                  method="gaussian", gaussian_kernel=gaussian_kernel, draw_boxes=draw_boxes)
            else:
                display_frame = apply_blur_to_image(display_frame, scaled_boxes, 
                                                  method="pixelate", pixel_blocks=pixel_blocks, draw_boxes=draw_boxes)
        else:
            # Nếu không detect được → vẫn ghi frame gốc (không bị đen màn hình)
            pass

        # Resize về kích thước output để file không quá nặng
        final_frame = cv2.resize(display_frame, (width, height))
        out_video.write(final_frame)

        processed_frames += 1
        progress_bar.progress(min(1.0, processed_frames / max(frame_count, 100)))

    cap.release()
    out_video.release()
    progress_bar.empty()
    return temp_output_path

# ---------------- MAIN LOGIC ----------------

if uploaded is not None:
    fname = uploaded.name.lower()
    is_image = any(fname.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".bmp"])
    is_video = any(fname.endswith(ext) for ext in [".mp4", ".mov", ".avi", ".mkv"])
    if is_image:
        st.info("Processing image...")
        result = process_image_file(uploaded)
        if result:
            out_img, boxes = result
            st.write(f"Detected {len(boxes)} faces.")
            st.image(cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB), channels="RGB", width="stretch", caption="Processed image")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Download processed image"):
                
                    _, im_buf_arr = cv2.imencode(".png", out_img)
                    st.download_button("Download PNG", im_buf_arr.tobytes(), file_name="processed.png", mime="image/png")
            with col2:
                if st.button("Unblur (restore original)"):
                    if st.session_state.original_img is not None:
                        st.image(cv2.cvtColor(st.session_state.original_img, cv2.COLOR_BGR2RGB), channels="RGB", width="stretch", caption="Original image")
                    else:
                        st.warning("Không có ảnh gốc trong phiên làm việc.")
    elif is_video:
        st.info("Received video. Processing may take a while depending on length.")
    
        t_in = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix)
        t_in.write(uploaded.read())
        t_in.flush()
        t_in.close()
        st.session_state.original_video_file = t_in.name
        t_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        t_out.close()
      
        processed_path = process_video_file(t_in.name, t_out.name)
        if processed_path:
            st.success("Hoàn tất xử lý video.")
            st.session_state.processed_video_file = processed_path
           
            st.video(processed_path)
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Download processed video"):
                    with open(processed_path, "rb") as f:
                        st.download_button("Download MP4", f.read(), file_name="processed.mp4", mime="video/mp4")
            with col2:
                if st.button("Unblur (play original)"):
                    if st.session_state.original_video_file:
                        st.video(st.session_state.original_video_file)
                    else:
                        st.warning("Không có video gốc trong phiên làm việc.")
        
    else:
        st.error("Định dạng file không được hỗ trợ.")

st.markdown("---")
st.write("### Notes / Tips")
st.markdown("""
- **Haar Cascade**: nhanh, dễ dùng, đôi khi bỏ sót faces nghiêng / trong điều kiện ánh sáng tệ.
- **DNN (res10 SSD)**: chính xác hơn, cần 2 file model (`deploy.prototxt` và `res10_300x300_ssd_iter_140000.caffemodel`). Nếu bạn muốn, tải 2 file đó từ nguồn OpenCV model zoo và nhập đường dẫn vào sidebar.
- **Video**: xử lý frame-by-frame, có thể chậm với video dài. Bạn có thể tối ưu bằng cách giảm độ phân giải khi detect, hoặc detect mỗi n frame rồi tracking (nâng cao).
- **Unblur**: ở đây mình lưu tạm bản gốc trong `st.session_state` để phục hồi khi người dùng nhấn `Unblur`. Lưu ý session_state không tồn tại qua nhiều phiên (browser restarts).
""")

st.write("If you want, mình có thể mở rộng: face tracking để blur liên tục trên video (giảm tần số detect), store results to folder, hoặc UI đẹp hơn.")
