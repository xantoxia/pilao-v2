import pandas as pd
import pickle
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import streamlit as st
from matplotlib import font_manager
import os
from openai import OpenAI
import base64
import requests
import datetime
import io
import pytz

# =============================
# GitHub配置
# =============================
GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
GITHUB_USERNAME = 'xantoxia'
GITHUB_REPO = 'blank-app-1'
GITHUB_BRANCH = 'main'
FILE_PATH = 'fatigue_data.csv'


# =============================
# 安全工具函数
# =============================
def safe_get(d, key, default=0):
    try:
        return d[key].values[0]
    except Exception:
        return default


def get_file_content(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except:
        return ""


def get_file_sha(file_path):
    url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{GITHUB_REPO}/contents/{file_path}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        return r.json().get("sha")
    return None


# =============================
# CSV保存
# =============================
def save_to_csv(input_data, result, body, cog, emo):
    tz = pytz.timezone("Asia/Shanghai")
    ts = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")

    data = {
        "颈部前屈": safe_get(input_data, "颈部前屈"),
        "颈部后仰": safe_get(input_data, "颈部后仰"),
        "肩部上举范围": safe_get(input_data, "肩部上举范围"),
        "肩部前伸范围": safe_get(input_data, "肩部前伸范围"),
        "肘部屈伸": safe_get(input_data, "肘部屈伸"),
        "手腕背伸": safe_get(input_data, "手腕背伸"),
        "手腕桡偏/尺偏": safe_get(input_data, "手腕桡偏/尺偏"),
        "背部屈曲范围": safe_get(input_data, "背部屈曲范围"),
        "持续时间": safe_get(input_data, "持续时间"),
        "重复频率": safe_get(input_data, "重复频率"),
        "fatigue_result": result,
        "body_score": calculate_score(body),
        "cog_score": calculate_score(cog),
        "emo_score": calculate_score(emo),
        "timestamp": ts
    }

    df = pd.DataFrame([data])

    if os.path.exists(FILE_PATH):
        try:
            old = pd.read_csv(FILE_PATH)
        except:
            old = pd.DataFrame()
        df = pd.concat([old, df], ignore_index=True)

    df.to_csv(FILE_PATH, index=False)


# =============================
# GitHub上传
# =============================
def upload_to_github(file_path):
    sha = get_file_sha(file_path)

    with open(file_path, "rb") as f:
        content = base64.b64encode(f.read()).decode()

    url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{GITHUB_REPO}/contents/{file_path}"

    data = {
        "message": "update fatigue data",
        "branch": GITHUB_BRANCH,
        "content": content
    }

    if sha:
        data["sha"] = sha

    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }

    r = requests.put(url, json=data, headers=headers)
    if r.status_code not in [200, 201]:
        st.error(r.json())


# =============================
# 评分函数（统一）
# =============================
def calculate_score(x):
    return {
        "请选择": 0,
        "完全没有": 1,
        "偶尔": 2,
        "经常": 3,
        "总是": 4
    }.get(x, 0)


# =============================
# AI调用（已彻底防崩）
# =============================
def call_ark_api(client, messages):
    try:
        completion = client.chat.completions.create(
            model="Pro/deepseek-ai/DeepSeek-V3.2",
            messages=messages,
            stream=True
        )

        for chunk in completion:
            try:
                content = getattr(chunk.choices[0].delta, "content", None)
                if content is None:
                    continue
                yield str(content)
            except:
                continue

    except Exception as e:
        yield f"[API_ERROR]{str(e)}"


# =============================
# 模型加载
# =============================
file_path = "corrected_fatigue_simulation_data_Chinese.csv"
data = pd.read_csv(file_path, encoding="gbk")

X = data.drop(columns=["疲劳等级"])
y = data["疲劳等级"]

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

@st.cache_resource
def load_model():
    return model


# =============================
# session_state 初始化（关键修复）
# =============================
for k, v in {
    "messages": [],
    "client": None,
    "ai_result": None,
    "result": None,
    "predictions": []
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# =============================
# UI
# =============================
st.title("疲劳评估系统")

col1, col2 = st.columns(2)

with col1:
    neck_flexion = st.slider("颈部前屈", 0, 60, 20)
    neck_extension = st.slider("颈部后仰", 0, 60, 25)
    shoulder_up = st.slider("肩部上举", 0, 180, 60)
    shoulder_forward = st.slider("肩部前伸", 0, 180, 120)

with col2:
    elbow = st.slider("肘部屈伸", 0, 180, 120)
    wrist_ext = st.slider("手腕背伸", 0, 60, 15)
    wrist_dev = st.slider("手腕偏转", 0, 30, 10)
    back = st.slider("背部屈曲", 0, 60, 20)

duration = st.number_input("持续时间", 0, 100)
freq = st.number_input("重复频率", 0, 100)

input_data = pd.DataFrame([{
    "颈部前屈": neck_flexion,
    "颈部后仰": neck_extension,
    "肩部上举范围": shoulder_up,
    "肩部前伸范围": shoulder_forward,
    "肘部屈伸": elbow,
    "手腕背伸": wrist_ext,
    "手腕桡偏/尺偏": wrist_dev,
    "背部屈曲范围": back,
    "持续时间": duration,
    "重复频率": freq
}])


# =============================
# 评估
# =============================
body = st.selectbox("身体疲劳", ["请选择","完全没有","偶尔","经常","总是"])
cog = st.selectbox("睡眠影响", ["请选择","完全没有","偶尔","经常","总是"])
emo = st.selectbox("肌肉酸痛", ["请选择","完全没有","偶尔","经常","总是"])


if st.button("评估"):
    if "请选择" in [body, cog, emo]:
        st.warning("请完整选择")
    else:
        pred = model.predict(input_data)[0]
        result = ["低疲劳","中疲劳","高疲劳"][pred]

        st.session_state.result = result

        save_to_csv(input_data, result, body, cog, emo)
        upload_to_github(FILE_PATH)

        st.success(result)


# =============================
# AI分析（修复核心崩溃点）
# =============================
if st.button("AI分析"):

    if st.session_state.result is None:
        st.warning("请先评估")
        st.stop()

    if st.session_state.client is None:
        st.session_state.client = OpenAI(
            api_key=st.secrets["API_KEY"],
            base_url="https://api.siliconflow.cn/v1"
        )

    prompt = f"""
用户疲劳：{st.session_state.result}
身体：{body} 睡眠：{cog} 肌肉：{emo}
角度：{input_data.to_dict()}

请分析人因风险并给建议
"""

    messages = [
        {"role": "system", "content": "你是人因工程专家"},
        {"role": "user", "content": prompt}
    ]

    response = ""

    with st.spinner("AI分析中..."):
        for chunk in call_ark_api(st.session_state.client, messages):

            if not chunk:
                continue

            if chunk.startswith("[API_ERROR]"):
                st.error(chunk)
                break

            response += chunk

    st.session_state.ai_result = response
    st.write(response)
