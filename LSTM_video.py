# LSTM_video.py (오류 수정 및 로직 개선 완료)

import cv2
import mediapipe as mp
import numpy as np
import torch
import time
import json
import os
import tensorflow as tf
from PIL import ImageFont, ImageDraw, Image

# --------------------------------------------------------------------------
#  1. 사용자 설정 부분
# --------------------------------------------------------------------------
VIDEO_FILE_PATH = "C:/Users/bit/Desktop/KakaoTalk_20250716_212034817.mp4" 
MODEL_DIR = "C:/Users/bit/Desktop"
# --------------------------------------------------------------------------


# 헬퍼 함수 1: 단어 모델을 위한 75차원 손 특징 생성
def get_75d_hand_features(joint):
    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :]
    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :]
    v = v2 - v1
    v = v / (np.linalg.norm(v, axis=1)[:, np.newaxis] + 1e-6)
    angle_v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :2]
    angle_v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :2]
    angle_v = angle_v2 - angle_v1
    angle_v = angle_v / (np.linalg.norm(angle_v, axis=1)[:, np.newaxis] + 1e-6)
    angle = np.degrees(np.arccos(np.einsum('nt,nt->n',
        angle_v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
        angle_v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])))
    return np.concatenate([v.flatten(), angle])

# 헬퍼 함수 2: 지문자 모델을 위한 55차원 손 특징 생성
def get_55d_hand_features(joint):
    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :2]
    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :2]
    v = v2 - v1
    v = v / (np.linalg.norm(v, axis=1)[:, np.newaxis] + 1e-6)
    angle = np.degrees(np.arccos(np.einsum('nt,nt->n',
        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])))
    return np.concatenate([v.flatten(), angle])

# Mediapipe 초기화
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose = mp_pose.Pose(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 동영상 파일 열기
cap = cv2.VideoCapture(VIDEO_FILE_PATH)
if not cap.isOpened():
    print(f"Error: Could not open video file: {VIDEO_FILE_PATH}")
    exit()

# --- 설정값 로드 및 변수 초기화 ---
model_path = os.path.join(MODEL_DIR, "lstm_sign_language_model_scripted.pt")
mean_path = os.path.join(MODEL_DIR, "data_mean.npy")
std_path = os.path.join(MODEL_DIR, "data_std.npy")
label_map_path = os.path.join(MODEL_DIR, "label_map.json")

#  인식률 향상을 위해 SEQ_LEN 값 복원 권장
SEQ_LEN = 60
OVERLAP_LEN = 40
FEATURE_DIM = 166

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device for video inference: {device}")

# 모델 로드 (단어/지문자)
try:
    model = torch.jit.load(model_path)
    model.eval()
    model.to(device)
    print(f"Word model loaded: {model_path}")
    
    tflite_model_path = "c:/Users/bit/Desktop/multi_hand_gesture_classifier.tflite"
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    alphabet_actions = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ','ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ','ㅐ', 'ㅒ', 'ㅔ', 'ㅖ', 'ㅢ', 'ㅚ', 'ㅟ']
    print(f"Fingerspelling model loaded: {tflite_model_path}")

    data_mean = np.load(mean_path)
    data_std = np.load(std_path)
    with open(label_map_path, 'r', encoding='utf-8') as f:
        labels_map_raw = json.load(f)
    labels_map = {v: k for k, v in labels_map_raw.items()}
    print("Auxiliary data loaded successfully.")
except Exception as e:
    print(f"Error during loading models or data files: {e}")
    exit()

# 버퍼 및 변수 초기화
keypoints_buffer, prediction_history = [], []
stable_prediction, stable_confidence = "Waiting...", 0.0
STABLE_THRESHOLD = 1
alphabet_seq_buffer, alphabet_prediction, alphabet_confidence = [], "...", 0.0
previous_alphabet_log = None
ALPHABET_SEQ_LEN = 10
previous_features, previous_velocity = None, None
idle_counter, IDLE_THRESHOLD, IDLE_TIME_THRESHOLD = 0, 0.5, 45

# 폰트 설정
try:
    font_path = "C:/Windows/Fonts/malgunbd.ttf"
    font = ImageFont.truetype(font_path, 30)
    font_small = ImageFont.truetype(font_path, 20)
except IOError:
    font = ImageFont.load_default()
    font_small = ImageFont.load_default()

frame_process_counter = 0
print("\n--- Starting video processing ---")

# 메인 루프
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video file reached.")
        break
    
    # 1. (수정) 좌우 반전 코드를 한 번만 실행하도록 수정
    frame = cv2.flip(frame, 1)

    frame_process_counter += 1
    frame_bgr = frame.copy()

    # 데이터 처리 (2프레임마다)
    if frame_process_counter % 2 == 0:
        frame_small = cv2.resize(frame, (480, 360), interpolation=cv2.INTER_AREA)
        frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)

        results_pose = pose.process(frame_rgb)
        results_hands = hands.process(frame_rgb)

        # 단어 모델용 특징 추출
        final_features = np.zeros(FEATURE_DIM)
        if results_pose.pose_landmarks:
            pose_lm = results_pose.pose_landmarks.landmark
            shoulder_center_x = (pose_lm[11].x + pose_lm[12].x) / 2
            shoulder_center_y = (pose_lm[11].y + pose_lm[12].y) / 2
            shoulder_width = np.linalg.norm([pose_lm[11].x - pose_lm[12].x, pose_lm[11].y - pose_lm[12].y]) + 1e-6
            pose_features = []
            for idx in [11, 12, 13, 14, 15, 16, 23, 24]:
                pose_features.append((pose_lm[idx].x - shoulder_center_x) / shoulder_width)
                pose_features.append((pose_lm[idx].y - shoulder_center_y) / shoulder_width)
            
            left_hand_features, right_hand_features = np.zeros(75), np.zeros(75)
            if results_hands.multi_hand_landmarks:
                for i, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
                    handedness = results_hands.multi_handedness[i].classification[0].label
                    joint = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                    hand_features_75d = get_75d_hand_features(joint)
                    if handedness == "Left": left_hand_features = hand_features_75d
                    elif handedness == "Right": right_hand_features = hand_features_75d
            final_features = np.concatenate([pose_features, left_hand_features, right_hand_features])

        velocity = np.zeros_like(final_features) if previous_features is None else final_features - previous_features
        acceleration = np.zeros_like(velocity) if previous_velocity is None else velocity - previous_velocity
        movement = np.sum(np.abs(velocity))
        idle_counter = idle_counter + 1 if movement < IDLE_THRESHOLD else 0
        if idle_counter >= IDLE_TIME_THRESHOLD:
            stable_prediction, stable_confidence, idle_counter = "Waiting...", 0.0, 0
            prediction_history.clear()
        
        combined_features = np.concatenate([final_features, velocity, acceleration])
        keypoints_buffer.append(combined_features)
        previous_features, previous_velocity = final_features, velocity

        #  2. (수정) 지문자 모델용 특징 추출 로직 개선
        is_right_hand_found = False
        if results_hands.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
                if results_hands.multi_handedness[i].classification[0].label == "Right":
                    is_right_hand_found = True
                    joint = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                    alphabet_feature_55d = get_55d_hand_features(joint)
                    alphabet_seq_buffer.append(alphabet_feature_55d)
                    break # 오른손을 찾았으면 루프 종료
        if not is_right_hand_found:
            alphabet_seq_buffer.clear()

    # 모델 예측 실행
    if len(keypoints_buffer) >= SEQ_LEN:
        input_tensor = torch.tensor((np.array(keypoints_buffer[-SEQ_LEN:]) - data_mean) / (data_std + 1e-8), dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        if confidence.item() > 0.7:
            predicted_label = labels_map.get(predicted_idx.item(), "Unknown")
            prediction_history.append(predicted_label)
            
            STABLE_THRESHOLD = 1 # 이 값도 3으로 맞춰주세요.
            if len(prediction_history) > STABLE_THRESHOLD:
                prediction_history.pop(0) # 가장 오래된 예측 제거
        
            if len(prediction_history) == STABLE_THRESHOLD and len(set(prediction_history)) == 1:
                if stable_prediction != prediction_history[0]:
                    stable_prediction = prediction_history[0]
                    stable_confidence = confidence.item() * 100
                    print(f"New stable prediction: {stable_prediction} ({stable_confidence:.2f}%)")
        del keypoints_buffer[:SEQ_LEN - OVERLAP_LEN]

    if len(alphabet_seq_buffer) >= ALPHABET_SEQ_LEN:
        input_data = np.expand_dims(np.array(alphabet_seq_buffer[-ALPHABET_SEQ_LEN:], dtype=np.float32), axis=0)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        y_pred = interpreter.get_tensor(output_details[0]['index'])
        i_pred = int(np.argmax(y_pred[0]))
        conf = y_pred[0][i_pred]
        if conf > 0.9:
            current_alphabet = alphabet_actions[i_pred]
            current_confidence = conf * 100

            # ✨ 터미널 로그 추가: 새로운 지문자가 인식될 때만 로그를 출력
            if current_alphabet != previous_alphabet_log:
                print(f"새로운 지문자 예측: {current_alphabet} ({current_confidence:.2f}%)")
                previous_alphabet_log = current_alphabet # 로그에 찍힌 지문자 업데이트
            
            alphabet_prediction, alphabet_confidence = current_alphabet, current_confidence

            if stable_prediction != "Waiting...": # 단어가 예측된 상태였다면 초기화
                print("지문자 인식되어 단어 버퍼를 초기화합니다.")
            keypoints_buffer.clear()
            prediction_history.clear()
            stable_prediction = "Waiting..."
            stable_confidence = 0.0
        else:
            alphabet_prediction, alphabet_confidence = "...", 0.0
            previous_alphabet_log = None # ✨ 인식 실패 시 로그 기록 리셋
        del alphabet_seq_buffer[0]

    # 시각화 및 화면 표시
    img_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    display_text = "인식 결과: ..."
    if alphabet_confidence > 90:
        display_text = f"인식 결과: {alphabet_prediction} (글자, {alphabet_confidence:.2f}%)"
    elif stable_prediction != "Waiting...":
        display_text = f"인식 결과: {stable_prediction} (단어, {stable_confidence:.2f}%)"
    draw.text((10, 30), display_text, font=font, fill=(255, 255, 0))
    
    text_frame_count = f"단어 버퍼: {len(keypoints_buffer)}/{SEQ_LEN} | 지문자 버퍼: {len(alphabet_seq_buffer)}/{ALPHABET_SEQ_LEN}"
    draw.text((10, 65), text_frame_count, font=font_small, fill=(255, 165, 0))
    
    frame_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    cv2.imshow('Sign Language Recognition from Video', frame_bgr)

    # 키보드 입력 ('q' 누르면 종료)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 종료 및 리소스 해제
print("--- Processing finished ---")
cap.release()
cv2.destroyAllWindows()