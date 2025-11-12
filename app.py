import streamlit as st
import cv2
import tempfile
import mediapipe as mp
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="捕球姿勢解析ツール", layout="wide")
st.title("🧤 捕球姿勢解析ツール（捕球真下原点版・安定版）")

# ----------------------------
# 1️⃣ 動画アップロード
# ----------------------------
video_file = st.file_uploader("🎞️ 動画ファイルをアップロード", type=["mp4", "mov", "avi"])

if video_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

    cap = cv2.VideoCapture(tfile.name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    st.sidebar.header("🎚️ フレーム操作")
    frame_num = st.sidebar.slider("フレーム番号", 0, total_frames - 1, 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()

    if not ret:
        st.error("⚠️ フレームの読み込みに失敗しました。")
    else:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, caption=f"Frame {frame_num}", use_container_width=True)

        # ----------------------------
        # 2️⃣ フレーム設定
        # ----------------------------
        if "ball_frame" not in st.session_state:
            st.session_state["ball_frame"] = None
            st.session_state["ball_xy"] = None
            st.session_state["catch_frame"] = None
            st.session_state["catch_xy"] = None

        col1, col2 = st.columns(2)
        with col1:
            if st.button("⚾ 打球出現フレーム設定"):
                st.session_state["ball_frame"] = frame_num
                st.session_state["ball_xy"] = None
                st.success(f"出現フレームを {frame_num} に設定しました！")
        with col2:
            if st.button("🧤 捕球フレーム設定"):
                st.session_state["catch_frame"] = frame_num
                st.session_state["catch_xy"] = None
                st.success(f"捕球フレームを {frame_num} に設定しました！")

        # ----------------------------
        # 3️⃣ 打球位置タップ
        # ----------------------------
        if st.session_state["ball_frame"] == frame_num and st.session_state.get("ball_xy") is None:
            st.info("👇 ボール位置をタップしてください")
            canvas_result = st_canvas(
                fill_color="",
                stroke_width=3,
                stroke_color="red",
                background_image=Image.fromarray(frame_rgb),
                update_streamlit=True,
                height=frame.shape[0],
                width=frame.shape[1],
                drawing_mode="point",
                key=f"canvas_ball_{frame_num}"  # フレーム番号で key を分ける
            )
            if canvas_result.json_data is not None:
                objects = canvas_result.json_data["objects"]
                if len(objects) > 0:
                    x, y = objects[-1]["left"], objects[-1]["top"]
                    st.session_state["ball_xy"] = (x, y)
                    st.success(f"ボール位置を取得: X={x:.1f}, Y={y:.1f}")

        # ----------------------------
        # 4️⃣ 捕球位置タップ
        # ----------------------------
        if st.session_state["catch_frame"] == frame_num and st.session_state.get("catch_xy") is None:
            st.info("👇 捕球位置をタップしてください")
            canvas_result = st_canvas(
                fill_color="",
                stroke_width=3,
                stroke_color="blue",
                background_image=Image.fromarray(frame_rgb),
                update_streamlit=True,
                height=frame.shape[0],
                width=frame.shape[1],
                drawing_mode="point",
                key=f"canvas_catch_{frame_num}"  # フレーム番号で key を分ける
            )
            if canvas_result.json_data is not None:
                objects = canvas_result.json_data["objects"]
                if len(objects) > 0:
                    x, y = objects[-1]["left"], objects[-1]["top"]
                    st.session_state["catch_xy"] = (x, y)
                    st.success(f"捕球位置を取得: X={x:.1f}, Y={y:.1f}")

        # ----------------------------
        # 5️⃣ 解析ボタン
        # ----------------------------
        if st.button("📊 捕球姿勢を解析"):
            if st.session_state.get("ball_xy") is None or st.session_state.get("catch_xy") is None:
                st.error("⚠️ 打球・捕球の座標をすべて指定してください")
            else:
                st.info("解析中... ⏳")

                cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state["catch_frame"])
                _, frame_catch = cap.read()
                frame_catch_rgb = cv2.cvtColor(frame_catch, cv2.COLOR_BGR2RGB)

                # Mediapipe Pose
                mp_pose = mp.solutions.pose
                pose = mp_pose.Pose(static_image_mode=True)
                results = pose.process(frame_catch_rgb)

                if not results.pose_landmarks:
                    st.error("⚠️ ポーズ検出に失敗しました。")
                else:
                    landmarks = results.pose_landmarks.landmark
                    lw = landmarks[15]  # 左手首

                    # 足首平均Yを地面として原点
                    foot_y = (landmarks[27].y + landmarks[28].y) / 2
                    origin = np.array([lw.x, foot_y, lw.z])

                    height_m = st.number_input("👤 身長 [m]", 1.0, 2.5, 1.75)
                    coords = []
                    for i, lm in enumerate(landmarks):
                        rel = np.array([lm.x, lm.y, lm.z]) - origin
                        rel_m = rel * height_m
                        coords.append((i, rel_m[0], -rel_m[1], rel_m[2]))

                    st.subheader("📊 捕球時の3D座標（捕球真下原点）")
                    for i, x, y, z in coords:
                        st.text(f"ID {i:02d}: X={x:.3f} m, Y={y:.3f} m, Z={z:.3f} m")

                    # 可視化
                    annotated = frame_catch_rgb.copy()
                    h, w, _ = annotated.shape
                    for lm in landmarks:
                        px, py = int(lm.x * w), int(lm.y * h)
                        cv2.circle(annotated, (px, py), 3, (0, 255, 0), -1)
                    cv2.circle(annotated,
                               (int(st.session_state["ball_xy"][0]), int(st.session_state["ball_xy"][1])),
                               6, (255, 0, 0), -1)
                    cv2.circle(annotated,
                               (int(st.session_state["catch_xy"][0]), int(st.session_state["catch_xy"][1])),
                               6, (0, 0, 255), -1)
                    st.image(annotated, caption="捕球姿勢＋打球位置（赤）＋捕球位置（青）", use_container_width=True)
