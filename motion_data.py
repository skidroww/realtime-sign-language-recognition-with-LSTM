# motion_data.py (수정 완료된 전체 코드)

import cv2
import numpy as np
import mediapipe as mp
import os
import time
from utils import Vector_Normalization

# Mediapipe 초기화
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose = mp_pose.Pose()
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) 
mp_drawing = mp.solutions.drawing_utils 

# 데이터 저장 경로 설정
BASE_PATH = 'C:\\Users\\bit\\Desktop\\sign_language_data'
os.makedirs(BASE_PATH, exist_ok=True)

# --- 메인 데이터 수집 및 변환 함수 ---
def process_sign_language_video(video_path, word, base_output_path=BASE_PATH): 
    print(f"Processing video: {video_path} for word: {word}")

    coord_output_folder = os.path.join(base_output_path, word, "좌표")
    os.makedirs(coord_output_folder, exist_ok=True)

    existing_files = [f for f in os.listdir(coord_output_folder) if f.startswith("keypoints_") and f.endswith(".npy")]  # 기존 파일 목록 가져오기
    next_sequence_number = 0
    if existing_files:
        numbers = [int(f.replace("keypoints_", "").replace(".npy", "")) for f in existing_files]
        next_sequence_number = max(numbers) + 1 if numbers else 0

    cap = cv2.VideoCapture(video_path) # 동영상 파일 열기
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}. Skipping.")
        return
    
    collected_data = [] # 수집된 데이터 저장 리스트
    previous_features = None  
    previous_velocity = None 
    FEATURE_DIM = 166  # 특징 벡터 차원 (포즈 48 + 왼손 75 + 오른손 75 + 속도 166 + 가속도 166)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_small = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA) # 동영상 프레임 크기 조정
        image = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB) # 색상 변환
        image.flags.writeable = False
        
        results_pose = pose.process(image)
        results_hands = hands.process(image)

        image_for_drawing = frame.copy()
        
        final_features = np.zeros(FEATURE_DIM)

        #  포즈가 감지되었을 때만 실행되는 블록
        if results_pose.pose_landmarks:
            # --- 특징 계산 로직 (기존과 동일) ---
            pose_lm = results_pose.pose_landmarks.landmark
            shoulder_center_x = (pose_lm[11].x + pose_lm[12].x) / 2 # 양쪽 어깨 중심
            shoulder_center_y = (pose_lm[11].y + pose_lm[12].y) / 2 # 양쪽 어깨 중심
            shoulder_width = np.linalg.norm(
                [pose_lm[11].x - pose_lm[12].x, pose_lm[11].y - pose_lm[12].y] # 어깨 너비 계산
            ) + 1e-6
            pose_indices = [11, 12, 13, 14, 15, 16, 23, 24] # 상체 관절 인덱스
            pose_features = []
            for idx in pose_indices:
                pose_features.append((pose_lm[idx].x - shoulder_center_x) / shoulder_width) # x 좌표 정규화
                pose_features.append((pose_lm[idx].y - shoulder_center_y) / shoulder_width) # y 좌표 정규화
            
            left_hand_features = np.zeros(75) # 왼손 특징 초기화
            right_hand_features = np.zeros(75) # 오른손 특징 초기화

            if results_hands.multi_hand_landmarks:
                for i, hand_landmarks in enumerate(results_hands.multi_hand_landmarks): # 손 랜드마크가 감지된 경우
                    handedness = results_hands.multi_handedness[i].classification[0].label 
                    joint = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])   # 손 관절 좌표 추출
                    v, angle_label = Vector_Normalization(joint) # 벡터 정규화 및 각도 레이블 계산
                    hand_features = np.concatenate([v.flatten(), angle_label.flatten()]) # 손 특징 벡터 생성
                    if handedness == "Left": left_hand_features = hand_features # 왼손 특징 저장
                    elif handedness == "Right": right_hand_features = hand_features # 오른손 특징 저장

            final_features = np.concatenate([pose_features, left_hand_features, right_hand_features]) # 최종 특징 벡터 생성

            # --- 미디어파이프 시각화 부분 ---
            mp_drawing.draw_landmarks(image_for_drawing, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            if results_hands.multi_hand_landmarks:
                for hand_landmarks in results_hands.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image_for_drawing, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        

        # --- 동적 특징 계산 로직 ---
        if previous_features is None:
            velocity = np.zeros_like(final_features) 
        else:
            velocity = final_features - previous_features # 이전 프레임과의 차이로 속도 계산

        if previous_velocity is None:
            acceleration = np.zeros_like(velocity)
        else:
            acceleration = velocity - previous_velocity # 이전 프레임과의 차이로 가속도 계산

        combined_features = np.concatenate([final_features, velocity, acceleration])
        collected_data.append(combined_features)

        previous_features = final_features
        previous_velocity = velocity

        # --- 최종 화면 표시 ---
        cv2.imshow("Sign Language Video Processing", image_for_drawing)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
            
    if collected_data:
        npy_file_name = f"keypoints_{next_sequence_number}.npy"
        npy_file_path = os.path.join(coord_output_folder, npy_file_name)
        np.save(npy_file_path, np.array(collected_data))
        print(f"Coordinates saved to {npy_file_path} ({len(collected_data)} frames, dim={np.array(collected_data).shape[1]}).")
    else:
        print(f"No pose data collected from {video_path}.")

    cap.release()
    cv2.destroyAllWindows()

# --- 동영상 파일 목록과 해당 수어 단어를 지정하여 함수 호출 ---
if __name__ == "__main__":
    
    words = ["어지럽다", "나", "쉬다", "어깨","감사","만나다","잘하다",
             "안녕하세요","무엇","비빔밥","기쁘다","취미","영화","얼굴",
             "보다","이름","같다","죄송","먹다","괜찮다","수고","나이",
             "다시","얼마","날","좋다","날짜","우리","전철","버스",
             "타다","휴대전화","어디","위치","책임","도착","가족","시간",
             "소개","주세요","물음","걷다","자매",
             "공부","사람","지금","어제","겨루다","당신",
             "결혼","노력","아니다","땀","아직","결과",
             "낳다","성공","서울","저녁","고객",
             "바라다"]
    
    
    base_data_dir = 'C:\\Users\\bit\\Desktop\\sign_language_data'

    all_video_files = []
    for word in words:
        word_folder = os.path.join(base_data_dir, f"{word}.영상")
        
        if os.path.exists(word_folder):
            for filename in os.listdir(word_folder):
                if filename.lower().endswith((".mp4", ".avi")):
                    video_path = os.path.join(word_folder, filename)
                    all_video_files.append({'path': video_path, 'word': word})
                    print(f"Found video: {video_path}")
        else:
            print(f"Warning: Word folder not found: {word_folder}")

    for file_info in all_video_files:
        process_sign_language_video(file_info['path'], file_info['word'])

    print("All video processing completed.")