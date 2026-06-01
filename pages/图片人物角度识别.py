import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image

# -------------- 关键修复：把模型初始化放到函数内，兼容Streamlit Cloud --------------
def load_models():
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    pose = mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8)
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    return mp_pose, mp_hands, pose, hands

# -------------- 工具函数 --------------
def get_coord(landmark, model_type='pose', img_width=640, img_height=480):
    """统一三维坐标处理（手部z轴补零）"""
    if model_type == 'pose':
        return [landmark.x * img_width, landmark.y * img_height, landmark.z * img_width]
    elif model_type == 'hands':
        return [landmark.x * img_width, landmark.y * img_height, 0]

def calculate_angle(a, b, c, plane='sagittal'):
    """安全的三维角度计算"""
    try:
        a = np.array(a)[:3].astype('float64')
        b = np.array(b)[:3].astype('float64')
        c = np.array(c)[:3].astype('float64')

        ba = a - b
        bc = c - b

        if plane == 'sagittal':
            ba = np.array([0, ba[1], ba[2]])
            bc = np.array([0, bc[1], bc[2]])
        elif plane == 'frontal':
            ba = np.array([ba[0], 0, ba[2]])
            bc = np.array([bc[0], 0, bc[2]])
        elif plane == 'transverse':
            ba = ba[:2]
            bc = bc[:2]

        ba_norm = np.linalg.norm(ba)
        bc_norm = np.linalg.norm(bc)
        if ba_norm < 1e-6 or bc_norm < 1e-6:
            return 0.0

        cosine = np.dot(ba, bc) / (ba_norm * bc_norm)
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    except:
        return 0.0

def calculate_neck_flexion(nose, shoulder_mid, hip_mid):
    try:
        nose = np.array(nose)[:2]
        shoulder_mid = np.array(shoulder_mid)[:2]
        hip_mid = np.array(hip_mid)[:2]
        torso_vector = hip_mid - shoulder_mid
        torso_angle = np.degrees(np.arctan2(torso_vector[1], torso_vector[0]))
        head_vector = nose - shoulder_mid
        head_angle = np.degrees(np.arctan2(head_vector[1], head_vector[0]))
        flexion_angle = head_angle - torso_angle
        if flexion_angle < 0:
            flexion_angle += 360
        if flexion_angle > 180:
            flexion_angle = 360 - flexion_angle
        return 180 - flexion_angle
    except:
        return 0.0

def calculate_trunk_flexion(shoulder_mid, hip_mid, knee_mid):
    try:
        torso_vector = np.array(hip_mid) - np.array(shoulder_mid)
        torso_angle = np.degrees(np.arctan2(torso_vector[1], torso_vector[0]))
        leg_vector = np.array(knee_mid) - np.array(hip_mid)
        leg_angle = np.degrees(np.arctan2(leg_vector[1], leg_vector[0]))
        flexion_angle = leg_angle - torso_angle
        if flexion_angle < 0:
            flexion_angle += 360
        if flexion_angle > 180:
            flexion_angle = 360 - flexion_angle
        return flexion_angle
    except:
        return 0.0

def draw_landmarks(image, joints):
    colors = {
        'neck': (255, 200, 0),
        'shoulder': (0, 255, 0),
        'elbow': (0, 255, 255),
        'wrist': (255, 0, 255)
    }
    nose = tuple(map(int, joints['鼻子'][:2]))
    shoulder_mid = tuple(map(int, joints['mid']['肩膀'][:2]))
    hip_mid = tuple(map(int, joints['mid']['臀部'][:2]))
    cv2.line(image, nose, shoulder_mid, colors['neck'], 2)
    cv2.line(image, shoulder_mid, hip_mid, colors['neck'], 2)
    for side in ['左侧', '右侧']:
        pt1 = tuple(map(int, joints[side]['肩膀'][:2]))
        pt2 = tuple(map(int, joints[side]['肘部'][:2]))
        cv2.line(image, pt1, pt2, colors['shoulder'], 2)
        pt3 = tuple(map(int, joints[side]['肘部'][:2]))
        pt4 = tuple(map(int, joints[side]['手腕'][:2]))
        cv2.line(image, pt3, pt4, colors['elbow'], 2)
        if '食指尖端' in joints[side]:
            pt5 = tuple(map(int, joints[side]['手腕'][:2]))
            pt6 = tuple(map(int, joints[side]['食指尖端'][:2]))
            cv2.line(image, pt5, pt6, colors['wrist'], 2)

def process_image(image):
    mp_pose, mp_hands, pose, hands = load_models()
    H, W, _ = image.shape
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pose_result = pose.process(img_rgb)
    hands_result = hands.process(img_rgb)
    metrics = {'angles': {}}

    if pose_result.pose_landmarks:
        def get_pose_pt(landmark):
            return get_coord(pose_result.pose_landmarks.landmark[landmark], 'pose', W, H)

        joints = {
            '左侧': {
                '肩膀': get_pose_pt(mp_pose.PoseLandmark.LEFT_SHOULDER),
                '肘部': get_pose_pt(mp_pose.PoseLandmark.LEFT_ELBOW),
                '手腕': get_pose_pt(mp_pose.PoseLandmark.LEFT_WRIST),
                '臀部': get_pose_pt(mp_pose.PoseLandmark.LEFT_HIP),
                '膝部': get_pose_pt(mp_pose.PoseLandmark.LEFT_KNEE)
            },
            '右侧': {
                '肩膀': get_pose_pt(mp_pose.PoseLandmark.RIGHT_SHOULDER),
                '肘部': get_pose_pt(mp_pose.PoseLandmark.RIGHT_ELBOW),
                '手腕': get_pose_pt(mp_pose.PoseLandmark.RIGHT_WRIST),
                '臀部': get_pose_pt(mp_pose.PoseLandmark.RIGHT_HIP),
                '膝部': get_pose_pt(mp_pose.PoseLandmark.RIGHT_KNEE)
            },
            'mid': {
                '肩膀': [(get_pose_pt(mp_pose.PoseLandmark.LEFT_SHOULDER)[i] +
                          get_pose_pt(mp_pose.PoseLandmark.RIGHT_SHOULDER)[i]) / 2 for i in range(3)],
                '臀部': [(get_pose_pt(mp_pose.PoseLandmark.LEFT_HIP)[i] +
                          get_pose_pt(mp_pose.PoseLandmark.RIGHT_HIP)[i]) / 2 for i in range(3)],
                '膝部': [(get_pose_pt(mp_pose.PoseLandmark.LEFT_KNEE)[i] +
                         get_pose_pt(mp_pose.PoseLandmark.RIGHT_KNEE)[i]) / 2 for i in range(3)]
            },
            '鼻子': get_pose_pt(mp_pose.PoseLandmark.NOSE)
        }

        if hands_result.multi_hand_landmarks:
            for hand in hands_result.multi_hand_landmarks:
                side = '左侧' if hand.landmark[0].x < 0.5 else '右侧'
                joints[side].update({
                    '手腕': get_coord(hand.landmark[mp_hands.HandLandmark.WRIST], 'hands', W, H),
                    '食指中节': get_coord(hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP], 'hands', W, H),
                    '食指尖端': get_coord(hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP], 'hands', W, H)
                })

        try:
            metrics['angles']['颈部前屈'] = calculate_neck_flexion(
                joints['鼻子'], joints['mid']['肩膀'], joints['mid']['臀部'])
            for side in ['左侧', '右侧']:
                metrics['angles'][f'{side} 肩部上举'] = calculate_angle(
                    joints[side]['臀部'], joints[side]['肩膀'], joints[side]['肘部'], 'frontal')
                metrics['angles'][f'{side} 肩部前伸'] = calculate_angle(
                    joints[side]['臀部'], joints[side]['肩膀'], joints[side]['肘部'], 'sagittal')
            for side in ['左侧', '右侧']:
                metrics['angles'][f'{side} 肘部屈伸'] = calculate_angle(
                    joints[side]['肩膀'], joints[side]['肘部'], joints[side]['手腕'], 'sagittal')
            for side in ['左侧', '右侧']:
                if '食指尖端' in joints[side]:
                    metrics['angles'][f'{side} 手腕背伸'] = calculate_angle(
                        joints[side]['肘部'], joints[side]['手腕'], joints[side]['食指尖端'], 'sagittal')
                    metrics['angles'][f'{side} 手腕桡偏'] = calculate_angle(
                        joints[side]['食指中节'], joints[side]['手腕'], joints[side]['食指尖端'], 'frontal')
            metrics['angles']['背部屈曲'] = calculate_trunk_flexion(
                joints['mid']['肩膀'], joints['mid']['臀部'], joints['mid']['膝部'])
            draw_landmarks(image, joints)
        except:
            pass
    pose.close()
    hands.close()
    return image, metrics

# ---------------------- Streamlit UI ----------------------
st.title("📸 角度分析系统")
st.markdown("""
**分析关节：**
- 颈部前屈
- 肩部上举/前伸
- 肘部屈伸
- 手腕背伸/桡偏
- 背部屈曲
""")
uploaded_file = st.file_uploader("上传工作场景图", type=["jpg", "png"])
threshold = st.slider("设置风险阈值(°)", 30, 90, 60)

if uploaded_file and uploaded_file.type.startswith("image"):
    img = Image.open(uploaded_file)
    img_np = np.array(img)
    if img_np.shape[-1] == 4:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
    else:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    with st.spinner("正在分析..."):
        processed_img, metrics = process_image(img_np)

    col1, col2 = st.columns(2)
    with col1:
        st.image(processed_img, channels="BGR", use_container_width=True)
    with col2:
        st.subheader("关节角度分析")
        for joint, angle in metrics['angles'].items():
            status = "⚠️" if angle > threshold else "✅"
            st.markdown(f"{status} **{joint}**: `{angle:.1f}°`")
else:
    st.info("请上传JPG/PNG格式的图片")
