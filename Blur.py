
import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from pathlib import Path

st.set_page_config(page_title="Face Blurring (Privacy Filter)", layout="centered")

@st.cache_resource
def load_haar():
 
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(cascade_path)

def load_dnn(prototxt_path: str, model_path: str):
  
    if not Path(prototxt_path).exists() or not Path(model_path).exists():
        raise FileNotFoundError("DNN model file(s) not found. Check paths.")
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    return net

def detect_faces_haar(image, face_cascade, scaleFactor=1.1, minNeighbors=5):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
    
    return rects

def detect_faces_dnn(image, net, conf_threshold=0.5):
    
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    boxes = []
    for i in range(0, detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
          
            startX = max(0, startX); startY = max(0, startY)
            endX = min(w - 1, endX); endY = min(h - 1, endY)
            boxes.append((startX, startY, endX - startX, endY - startY))
    return boxes

def gaussian_blur_region(image, rect, ksize=(51, 51)):
    x, y, w, h = rect
    sub = image[y:y+h, x:x+w]
    
    kx = ksize[0] if ksize[0] % 2 == 1 else ksize[0] + 1
    ky = ksize[1] if ksize[1] % 2 == 1 else ksize[1] + 1
    kx = min(kx, max(1, w//2*2+1))
    ky = min(ky, max(1, h//2*2+1))
    blurred = cv2.GaussianBlur(sub, (kx, ky), 0)
    out = image.copy()
    out[y:y+h, x:x+w] = blurred
    return out

def pixelate_region(image, rect, blocks=10):
    x, y, w, h = rect
    sub = image[y:y+h, x:x+w]
    (bh, bw) = (max(1, blocks), max(1, blocks))
    
    small = cv2.resize(sub, (bw, bh), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    out = image.copy()
    out[y:y+h, x:x+w] = pixelated
    return out

def apply_blur_to_image(image, boxes, method="gaussian", gaussian_kernel=(51,51), pixel_blocks=10, draw_boxes=False):
    out = image.copy()
    for rect in boxes:
        if method == "gaussian":
            out = gaussian_blur_region(out, rect, ksize=gaussian_kernel)
        elif method == "pixelate":
            out = pixelate_region(out, rect, blocks=pixel_blocks)
        if draw_boxes:
            x, y, w, h = rect
            cv2.rectangle(out, (x, y), (x+w, y+h), (0,255,0), 2)
    return out


st.title("🔒 Face Blurring — Privacy Filter")
st.write("Upload an image or video. Detect faces (Haar / DNN) and Blur (Gaussian or Pixelate).")

with st.sidebar:
    st.header("Settings")
    detector = st.selectbox("Face detector", ("Haar Cascade (fast)", "DNN (more accurate)"))
    blur_method = st.selectbox("Blur method", ("Gaussian", "Pixelate"))
    draw_boxes = st.checkbox("Draw detection boxes (for debug)", value=False)
    if blur_method == "Gaussian":
        k = st.slider("Gaussian kernel size (odd)", 11, 101, 51, step=2)
        gaussian_kernel = (k, k)
    else:
        pixel_blocks = st.slider("Pixelate blocks (lower = stronger)", 2, 40, 10)

    st.markdown("---")
    st.write("DNN model (optional):")
    prototxt_path = st.text_input("Prototxt path (e.g. deploy.prototxt)", "")
    model_path = st.text_input("Caffe model path (e.g. res10_300x300_ssd_iter_140000.caffemodel)", "")
    st.info("If you choose DNN, provide the two files above. If empty, Haar will be used.")

st.write("### 1) Upload file")
uploaded = st.file_uploader("Choose an image or video", type=["jpg","jpeg","png","bmp","mp4","mov","avi","mkv"], accept_multiple_files=False)

if "original_img" not in st.session_state:
    st.session_state.original_img = None
if "processed_img" not in st.session_state:
    st.session_state.processed_img = None
if "original_video_file" not in st.session_state:
    st.session_state.original_video_file = None
if "processed_video_file" not in st.session_state:
    st.session_state.processed_video_file = None


face_cascade = load_haar()
dnn_net = None
if detector.startswith("DNN") and prototxt_path and model_path:
    try:
        dnn_net = load_dnn(prototxt_path, model_path)
    except Exception as e:
        st.error(f"Failed to load DNN: {e}")
        dnn_net = None


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

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  
    fps = cap.get(cv2.CAP_PROP_FPS) 
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_video = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
       
        if detector.startswith("DNN") and dnn_net is not None:
            boxes = detect_faces_dnn(frame, dnn_net, conf_threshold=0.5)
        else:
            boxes = detect_faces_haar(frame, face_cascade)
        if blur_method == "Gaussian":
            frame_out = apply_blur_to_image(frame, boxes, method="gaussian", gaussian_kernel=gaussian_kernel, draw_boxes=draw_boxes)
        else:
            frame_out = apply_blur_to_image(frame, boxes, method="pixelate", pixel_blocks=pixel_blocks, draw_boxes=draw_boxes)
        out_video.write(frame_out)
        i += 1
        if frame_count > 0:
            progress_bar.progress(min(1.0, i / frame_count))
    cap.release()
    out_video.release()
    cv2.destroyAllWindows()
    progress_bar.empty()
    if os.path.exists(temp_input_path) and os.path.getsize(temp_input_path) > 0:
        return temp_input_path
    else:
        st.error("Xử lý video thất bại.")
        return None


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
            st.image(cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB), channels="RGB", use_container_stretch=True, caption="Processed image")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Download processed image"):
                
                    _, im_buf_arr = cv2.imencode(".png", out_img)
                    st.download_button("Download PNG", im_buf_arr.tobytes(), file_name="processed.png", mime="image/png")
            with col2:
                if st.button("Unblur (restore original)"):
                    if st.session_state.original_img is not None:
                        st.image(cv2.cvtColor(st.session_state.original_img, cv2.COLOR_BGR2RGB), channels="RGB", use_container_stretch=True, caption="Original image")
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
