import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.title("📸 图片人体角度识别")
uploaded_file = st.file_uploader("上传图片", type=["png", "jpg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    arr = np.array(img)
    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    # 只画个框做演示，先保证页面不报错
    h, w = arr.shape[:2]
    cv2.rectangle(arr, (w//4, h//4), (3*w//4, 3*h//4), (0, 255, 0), 3)

    col1, col2 = st.columns(2)
    with col1:
        st.image(arr, channels="BGR", caption="识别结果（演示版）")
    with col2:
        st.subheader("角度数据（演示版）")
        st.success("颈部前屈: 20.0°")
        st.success("左肩角度: 60.0°")
        st.success("右肩角度: 60.0°")
