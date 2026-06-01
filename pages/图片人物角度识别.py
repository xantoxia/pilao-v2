import cv2
import numpy as np
import streamlit as st
from PIL import Image

# -------------- 关键修复：不在全局加载 MediaPipe，只在需要时加载 --------------
def analyze_image(img_np):
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands

    pose = mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8)
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

    H, W, _ = img_np.shape
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    pose_result = pose.process(img_rgb)
    hands_result = hands.process(img_rgb)
    metrics = {'angles': {}}

    def get_coord(lm, w, h):
        return [lm.x * w, lm.y * h, lm.z * w]

    def ang(a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        ba = a - b
        bc = c - b
        cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

    if pose_result.pose_landmarks:
        lm = pose_result.pose_landmarks.landmark

        nose = get_coord(lm[mp_pose.PoseLandmark.NOSE], W, H)
        l_sh = get_coord(lm[mp_pose.PoseLandmark.LEFT_SHOULDER], W, H)
        r_sh = get_coord(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER], W, H)
        l_el = get_coord(lm[mp_pose.PoseLandmark.LEFT_ELBOW], W, H)
        r_el = get_coord(lm[mp_pose.PoseLandmark.RIGHT_ELBOW], W, H)
        l_hi = get_coord(lm[mp_pose.PoseLandmark.LEFT_HIP], W, H)
        r_hi = get_coord(lm[mp_pose.PoseLandmark.RIGHT_HIP], W, H)

        sh_mid = [(l_sh[i]+r_sh[i])/2 for i in range(3)]
        hi_mid = [(l_hi[i]+r_hi[i])/2 for i in range(3)]

        # 计算角度
        try:
            metrics['angles']['颈部前屈'] = ang(hi_mid, sh_mid, nose)
            metrics['angles']['左肩角度'] = ang(l_hi, l_sh, l_el)
            metrics['angles']['右肩角度'] = ang(r_hi, r_sh, r_el)
        except:
            pass

        # 简单画线
        cv2.line(img_np,
                 (int(nose[0]), int(nose[1])),
                 (int(sh_mid[0]), int(sh_mid[1])),
                 (0, 255, 0), 3)

    pose.close()
    hands.close()
    return img_np, metrics

# -------------------- Streamlit 界面 --------------------
st.title("📸 图片人体角度识别")
upload = st.file_uploader("上传图片", type=["png","jpg"])

if upload:
    img = Image.open(upload).convert("RGB")
    arr = np.array(img)
    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    with st.spinner("识别中..."):
        out_img, angles = analyze_image(arr)

    c1, c2 = st.columns(2)
    with c1:
        st.image(out_img, channels="BGR", caption="识别结果")
    with c2:
        st.subheader("角度数据")
        for k, v in angles['angles'].items():
            st.success(f"{k}: {v:.1f}°")
