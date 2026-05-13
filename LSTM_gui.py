# LSTM_gui.py (수정 완료된 전체 코드)

import cv2
import mediapipe as mp
import numpy as np
import torch
import time
import json
import os
import tensorflow as tf

from PIL import ImageFont, ImageDraw, Image
from utils import Vector_Normalization 


# 헬퍼 함수 1: 단어 모델을 위한 75차원 손 특징 생성
def get_75d_hand_features(joint):
    # Z축 포함하여 75개 특징 생성 (벡터 60개 + 각도 15개)
    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :] # Z축 포함 (:3 과 동일)
    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :] # Z축 포함
    v = v2 - v1
    v = v / (np.linalg.norm(v, axis=1)[:, np.newaxis] + 1e-6)
    
    # 각도는 X, Y만으로 계산
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
    # X, Y축만 사용하여 55개 특징 생성 (벡터 40개 + 각도 15개)
    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :2] # X, Y만 사용
    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :2] # X, Y만 사용
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
mp_drawing = mp.solutions.drawing_utils

# 웹캠 또는 IP 카메라 스트림 열기
cap = cv2.VideoCapture('http://10.101.171.170:4747/video')

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# --- 설정값 로드 ---
MODEL_DIR = "C:/Users/bit/Desktop"
model_path = os.path.join(MODEL_DIR, "lstm_sign_language_model_scripted.pt")
mean_path = os.path.join(MODEL_DIR, "data_mean.npy")
std_path = os.path.join(MODEL_DIR, "data_std.npy")
label_map_path = os.path.join(MODEL_DIR, "label_map.json")

SEQ_LEN = 60
OVERLAP_LEN = 40

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device for GUI inference: {device}")

# 모델 로드
try:
    model = torch.jit.load(model_path)
    model.eval()
    model.to(device)
    print(f"Model loaded successfully from {model_path} and moved to {device}.")
except Exception as e:
    print(f"Error loading model from {model_path}: {e}")
    exit()
    
    
    
# --- ▼▼▼ 지문자 인식 모델 로드 코드 추가 ▼▼▼ ---
try:
    # 지문자 모델 파일 경로 (실제 경로에 맞게 수정하세요)
    tflite_model_path = "c:/Users/bit/Desktop/multi_hand_gesture_classifier.tflite"
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # 지문자 리스트
    alphabet_actions = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
                        'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ',
                        'ㅐ', 'ㅒ', 'ㅔ', 'ㅖ', 'ㅢ', 'ㅚ', 'ㅟ']
                        
    print("지문자 인식 모델(TFLite) 로드 완료.")
except Exception as e:
    print(f"TFLite 모델 로드 오류: {e}")
    exit()
    
    

# 표준화 파라미터 로드
try:
    data_mean = np.load(mean_path)
    data_std = np.load(std_path)
    print("Standardization mean and std loaded successfully.")
except Exception as e:
    print(f"Error loading standardization parameters: {e}")
    exit()

# 라벨 맵 로드
try:
    with open(label_map_path, 'r', encoding='utf-8') as f:
        labels_map_raw = json.load(f)
    labels_map = {v: k for k, v in labels_map_raw.items()}
    print("Label map loaded successfully.")
except Exception as e:
    print(f"Error loading label map from {label_map_path}: {e}")
    exit()

# 버퍼 및 예측 관련 변수 초기화
keypoints_buffer = []
prediction_history = [] # 최근 예측 단어들을 저장할 리스트
STABLE_THRESHOLD = 3 # 몇 번 연속으로 일치해야 인정할지 결정 (3번)
stable_prediction = "Waiting..." # 안정화된 최종 예측 결과
stable_confidence = 0.0 
current_prediction = "Waiting..."
current_confidence = 0.0
current_probabilities = {label: 0.0 for label in labels_map.values()}

# --- ▼▼▼ 지문자 모델용 버퍼 및 변수 추가 ▼▼▼ ---
alphabet_seq_buffer = []
alphabet_prediction = "..."
alphabet_confidence = 0.0
ALPHABET_SEQ_LEN = 10 # 지문자 모델은 10프레임만 필요

# --- 👇 유휴 상태 감지용 변수 추가 ---
IDLE_THRESHOLD = 0.5  # 움직임 감지를 위한 임계값. 이 값보다 작으면 정지 상태로 간주.
IDLE_TIME_THRESHOLD = 45 # 유휴 상태로 판단하기 위한 프레임 수 (약 1.5초, 30fps 기준)
idle_counter = 0      # 정지 상태가 지속된 프레임 수를 세는 카운터

# --- 💡 수정/추가된 부분 시작 ---
# 동적 특징 계산을 위한 변수 추가
previous_features = None
previous_velocity = None
FEATURE_DIM = 166 # 원본 특징(좌표)의 차원
# --- 💡 수정/추가된 부분 끝 ---

# 폰트 설정
try:
    font_path = "C:/Windows/Fonts/malgunbd.ttf"
    font = ImageFont.truetype(font_path, 30)
    font_small = ImageFont.truetype(font_path, 20)
except IOError:
    print("Error: Could not load font. Using default.")
    font = ImageFont.load_default()
    font_small = ImageFont.load_default()

# --- 프레임 처리 관련 변수 ---
frame_process_counter = 0
last_results_pose = None
last_results_hands = None
start_time = time.time()
fps_frame_count = 0

# LSTM_gui.py 파일의 while 루프 전체를 아래 코드로 교체하세요.

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break
    
    frame = cv2.flip(frame, 1)  # 1은 좌우 반전을 의미합니다.

    frame_process_counter += 1
    frame_bgr = frame.copy()

    # --------------------------------------------------------------------------
    # 1. 데이터 처리 및 예측 (2프레임마다 한 번씩 실행)
    # --------------------------------------------------------------------------
    if frame_process_counter % 2 == 0:
        frame_small = cv2.resize(frame, (480, 360), interpolation=cv2.INTER_AREA)
        frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)

        results_pose = pose.process(frame_rgb)
        results_hands = hands.process(frame_rgb)
        
        # 시각화를 위해 마지막 감지 결과 저장
        last_results_pose = results_pose
        last_results_hands = results_hands

        # --- 특징 추출 (단어 모델용) ---
        final_features = np.zeros(FEATURE_DIM)
        if results_pose.pose_landmarks:
            pose_lm = results_pose.pose_landmarks.landmark
            shoulder_center_x = (pose_lm[11].x + pose_lm[12].x) / 2
            shoulder_center_y = (pose_lm[11].y + pose_lm[12].y) / 2
            shoulder_width = np.linalg.norm([pose_lm[11].x - pose_lm[12].x, pose_lm[11].y - pose_lm[12].y]) + 1e-6
            
            pose_indices = [11, 12, 13, 14, 15, 16, 23, 24]
            pose_features = []
            for idx in pose_indices:
                pose_features.append((pose_lm[idx].x - shoulder_center_x) / shoulder_width)
                pose_features.append((pose_lm[idx].y - shoulder_center_y) / shoulder_width)
            
            left_hand_features = np.zeros(75)
            right_hand_features = np.zeros(75)
            
            if results_hands.multi_hand_landmarks:
                for i, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
                    handedness = results_hands.multi_handedness[i].classification[0].label
                    joint = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                    #v, angle_label = Vector_Normalization(joint)
                    #hand_features = np.concatenate([v.flatten(), angle_label.flatten()])
                    
                    hand_features_75d = get_75d_hand_features(joint)
                    
                    if handedness == "Left":
                        left_hand_features = hand_features_75d
                    elif handedness == "Right":
                        right_hand_features = hand_features_75d

            final_features = np.concatenate([pose_features, left_hand_features, right_hand_features])

        velocity = np.zeros_like(final_features) if previous_features is None else final_features - previous_features
        acceleration = np.zeros_like(velocity) if previous_velocity is None else velocity - previous_velocity
        
        movement = np.sum(np.abs(velocity))
        if movement < IDLE_THRESHOLD:
            idle_counter += 1
        else:
            idle_counter = 0

        if idle_counter >= IDLE_TIME_THRESHOLD:
            stable_prediction = "Waiting..."
            stable_confidence = 0.0
            prediction_history.clear()
            idle_counter = 0
            
        combined_features = np.concatenate([final_features, velocity, acceleration])
        keypoints_buffer.append(combined_features)
        
        previous_features = final_features
        previous_velocity = velocity

        # --- 특징 추출 (지문자 모델용) ---
        is_right_hand_found = False
        if results_hands.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
                if results_hands.multi_handedness[i].classification[0].label == "Right":
                    is_right_hand_found = True
                    joint = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                    #v, angle_label = Vector_Normalization(joint)
                    
                    alphabet_feature_55d = get_55d_hand_features(joint)
                    alphabet_seq_buffer.append(alphabet_feature_55d)
                    break
        if not is_right_hand_found:
            alphabet_seq_buffer.clear()

    # --------------------------------------------------------------------------
    # 2. 모델 예측 실행 (매 프레임마다 버퍼를 확인하여 독립적으로 실행)
    # --------------------------------------------------------------------------
    
    # --- 단어 모델 예측 ---
    if len(keypoints_buffer) >= SEQ_LEN:
        sequence_to_predict = np.array(keypoints_buffer[-SEQ_LEN:])
        sequence_standardized = (sequence_to_predict - data_mean) / (data_std + 1e-8)
        input_tensor = torch.tensor(sequence_standardized, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        if confidence.item() > 0.7:
            predicted_label = labels_map.get(predicted_idx.item(), "Unknown")
            prediction_history.append(predicted_label)
            if len(prediction_history) > STABLE_THRESHOLD:
                prediction_history.pop(0)
            if len(prediction_history) == STABLE_THRESHOLD and len(set(prediction_history)) == 1:
                if stable_prediction != prediction_history[0]:
                    stable_prediction = prediction_history[0]
                    stable_confidence = confidence.item() * 100
        del keypoints_buffer[:SEQ_LEN - OVERLAP_LEN]

    # --- 지문자 모델 예측 ---
    if len(alphabet_seq_buffer) >= ALPHABET_SEQ_LEN:
        input_data = np.expand_dims(np.array(alphabet_seq_buffer[-ALPHABET_SEQ_LEN:], dtype=np.float32), axis=0)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        y_pred = interpreter.get_tensor(output_details[0]['index'])
        i_pred = int(np.argmax(y_pred[0]))
        conf = y_pred[0][i_pred]
        if conf > 0.9:
            alphabet_prediction = alphabet_actions[i_pred]
            alphabet_confidence = conf * 100
        else:
            alphabet_prediction = "..."
            alphabet_confidence = 0.0
        del alphabet_seq_buffer[0]

    # --------------------------------------------------------------------------
    # 3. 시각화 및 화면 표시 (매 프레임마다 실행)
    # --------------------------------------------------------------------------
    img_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    
    display_text = ""
    if alphabet_confidence > 90:
        display_text = f"인식 결과: {alphabet_prediction} (글자, {alphabet_confidence:.2f}%)"
    elif stable_prediction != "Waiting...":
        display_text = f"인식 결과: {stable_prediction} (단어, {stable_confidence:.2f}%)"
    else:
        display_text = "인식 결과: ..."
    draw.text((10, 30), display_text, font=font, fill=(255, 255, 0))
    
    text_frame_count = f"단어 버퍼: {len(keypoints_buffer)}/{SEQ_LEN} | 지문자 버퍼: {len(alphabet_seq_buffer)}/{ALPHABET_SEQ_LEN}"
    draw.text((10, 65), text_frame_count, font=font_small, fill=(255, 165, 0))
    
    frame_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    cv2.imshow('Sign Language Recognition', frame_bgr)
    cases = []
    
    # --------------------------------------------------------------------------
    # 4. 키보드 입력 처리
    # --------------------------------------------------------------------------
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        print("Prediction state has been reset by user.")
        keypoints_buffer.clear()
        prediction_history.clear()
        alphabet_seq_buffer.clear() # 지문자 버퍼도 초기화
        stable_prediction = "Waiting..."
        stable_confidence = 0.0
        alphabet_prediction = "..."
        alphabet_confidence = 0.0
        idle_counter = 0

cap.release()
cv2.destroyAllWindows()