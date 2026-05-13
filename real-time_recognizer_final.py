# real-time_recognizer_final.py

import cv2
import mediapipe as mp
import numpy as np
import torch
import tensorflow as tf
import json
import os
from collections import deque
from PIL import ImageFont, ImageDraw, Image

class FeatureExtractor:
    """
    영상 프레임에서 수어 인식을 위한 포즈 및 손 특징점을 추출합니다.
    단어 및 지문자 모델에 필요한 특징을 계산하고 움직임을 감지합니다.
    (LSTM_video_OOP2.py와 동일한 클래스)
    """
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.pose = self.mp_pose.Pose(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.8)
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.8)
        self.previous_features = None
        self.previous_velocity = None

    def _get_75d_hand_features(self, joint):
        """단어 모델을 위한 75차원 손 특징을 생성하는 헬퍼 함수."""
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

    def _get_55d_hand_features(self, joint):
        """지문자 모델을 위한 55차원 손 특징을 생성하는 헬퍼 함수."""
        v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :2]
        v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :2]
        v = v2 - v1
        v = v / (np.linalg.norm(v, axis=1)[:, np.newaxis] + 1e-6)
        angle = np.degrees(np.arccos(np.einsum('nt,nt->n',
            v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:],
            v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])))
        return np.concatenate([v.flatten(), angle])

    def extract(self, frame):
        """한 프레임을 처리하여 필요한 모든 특징을 추출합니다."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_pose = self.pose.process(frame_rgb)
        results_hands = self.hands.process(frame_rgb)
        
        num_hands_detected = 0
        if results_hands.multi_hand_landmarks:
            num_hands_detected = len(results_hands.multi_hand_landmarks)
        
        pose_features = np.zeros(16)
        word_model_features = np.zeros(166) # 16 (pose) + 75 (left) + 75 (right)
        
        if results_pose.pose_landmarks:
            pose_lm = results_pose.pose_landmarks.landmark
            shoulder_center_x = (pose_lm[11].x + pose_lm[12].x) / 2
            shoulder_center_y = (pose_lm[11].y + pose_lm[12].y) / 2
            shoulder_width = np.linalg.norm([pose_lm[11].x - pose_lm[12].x, pose_lm[11].y - pose_lm[12].y]) + 1e-6
            temp_pose_features = []
            for idx in [11, 12, 13, 14, 15, 16, 23, 24]:
                temp_pose_features.append((pose_lm[idx].x - shoulder_center_x) / shoulder_width)
                temp_pose_features.append((pose_lm[idx].y - shoulder_center_y) / shoulder_width)
            pose_features = np.array(temp_pose_features)

        left_hand_features, right_hand_features = np.zeros(75), np.zeros(75)
        alphabet_feature = None
        if results_hands.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
                handedness = results_hands.multi_handedness[i].classification[0].label
                joint = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                hand_features_75d = self._get_75d_hand_features(joint)
                if handedness == "Left":
                    left_hand_features = hand_features_75d
                elif handedness == "Right":
                    right_hand_features = hand_features_75d
                    alphabet_feature = self._get_55d_hand_features(joint)

        word_model_features = np.concatenate([pose_features, left_hand_features, right_hand_features])
        
        if self.previous_features is None: self.previous_features = np.zeros_like(word_model_features)
        velocity = word_model_features - self.previous_features
        if self.previous_velocity is None: self.previous_velocity = np.zeros_like(velocity)
        acceleration = velocity - self.previous_velocity
        
        movement = np.sum(np.abs(velocity))
        self.previous_features = word_model_features
        self.previous_velocity = velocity

        dynamic_word_features = np.concatenate([word_model_features, velocity, acceleration])
        
        return dynamic_word_features, alphabet_feature, movement, results_pose, results_hands, num_hands_detected

    def close(self):
        """Mediapipe 리소스를 해제합니다."""
        self.pose.close()
        self.hands.close()

class Predictor:
    """
    단어(PyTorch) 및 지문자(TFLite) 모델을 로드하고 예측을 관리합니다.
    (LSTM_video_OOP2.py와 동일한 클래스)
    """
    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config

        # --- 단어 모델 (PyTorch) ---
        self.word_model = torch.jit.load(os.path.join(config['MODEL_DIR'], "lstm_sign_language_model_scripted.pt")).to(self.device)
        self.word_model.eval()
        self.data_mean = np.load(os.path.join(config['MODEL_DIR'], "data_mean.npy"))
        self.data_std = np.load(os.path.join(config['MODEL_DIR'], "data_std.npy"))
        with open(os.path.join(config['MODEL_DIR'], "label_map.json"), 'r', encoding='utf-8') as f:
            self.word_labels_map = {v: k for k, v in json.load(f).items()}
        
        # --- 지문자 모델 (TFLite) ---
        tflite_path = os.path.join(config['MODEL_DIR'], "multi_hand_gesture_classifier.tflite")
        self.alphabet_interpreter = tf.lite.Interpreter(model_path=tflite_path)
        self.alphabet_interpreter.allocate_tensors()
        self.alphabet_input_details = self.alphabet_interpreter.get_input_details()
        self.alphabet_output_details = self.alphabet_interpreter.get_output_details()
        self.alphabet_actions = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ','ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ','ㅐ', 'ㅒ', 'ㅔ', 'ㅖ', 'ㅢ', 'ㅚ', 'ㅟ']

        # --- 버퍼 및 파라미터 ---
        self.word_buffer = deque()
        self.word_history = []
        self.alphabet_buffer = deque(maxlen=config['SEQ_LEN_ALPHABET'])
        self.alphabet_confirm_buffer = deque(maxlen=5)
        self.last_confirmed_alphabet = None

    def predict_word(self, features):
        """
        특징(features) 시퀀스로부터 단어를 예측합니다.
        '호핑 윈도우'와 '안정성 검사'를 모두 포함한 원본 로직입니다.
        """
        # 1. 버퍼에 새로운 특징 추가
        self.word_buffer.append(features)

        # 2. 버퍼가 예측에 필요한 최소 길이에 도달했는지 확인
        if len(self.word_buffer) < self.config['SEQ_LEN_WORD']:
            return None, 0.0

        # 3. 예측 수행
        sequence = np.array(list(self.word_buffer))
        
        normalized_sequence = (sequence - self.data_mean) / (self.data_std + 1e-8)
        input_tensor = torch.tensor(normalized_sequence, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probabilities = torch.softmax(self.word_model(input_tensor), dim=1)
        
        confidence, idx = torch.max(probabilities, 1)
        confidence_item = confidence.item()
        
        # 4. 호핑 윈도우: 예측 후 버퍼에서 오래된 데이터 제거
        hop_size = self.config['SEQ_LEN_WORD'] - self.config['OVERLAP_LEN_WORD']
        for _ in range(hop_size):
            if self.word_buffer:
                self.word_buffer.popleft()

        # 5. 신뢰도 임계값 확인
        if confidence_item < self.config['CONF_THRESHOLD_WORD']:
            return None, 0.0
        
        # 6. 예측 안정화 로직 (원본과 동일하게 복원)
        label = self.word_labels_map.get(idx.item(), "Unknown")
        self.word_history.append(label)

        if len(self.word_history) > self.config['STABLE_THRESHOLD_WORD']:
            self.word_history.pop(0)

        # 안정화 조건 확인: 버퍼가 꽉 찼고, 모든 요소가 동일한가?
        if (len(self.word_history) == self.config['STABLE_THRESHOLD_WORD'] and
                len(set(self.word_history)) == 1):
            return self.word_history[0], confidence_item
        
        return None, 0.0

    def predict_fingerspelling(self, features):
        """디바운싱으로 안정화된 지문자를 예측합니다."""
        if features is None:
            self.alphabet_confirm_buffer.append(None)
            return None, 0.0

        self.alphabet_buffer.append(features)
        if len(self.alphabet_buffer) < self.config['SEQ_LEN_ALPHABET']:
            return None, 0.0

        input_data = np.expand_dims(np.array(self.alphabet_buffer, dtype=np.float32), axis=0)
        self.alphabet_interpreter.set_tensor(self.alphabet_input_details[0]['index'], input_data)
        self.alphabet_interpreter.invoke()
        y_pred = self.alphabet_interpreter.get_tensor(self.alphabet_output_details[0]['index'])
        i_pred = int(np.argmax(y_pred[0]))
        confidence = y_pred[0][i_pred]
        
        if confidence > self.config['CONF_THRESHOLD_ALPHABET']:
            self.alphabet_confirm_buffer.append(self.alphabet_actions[i_pred])
        else:
            self.alphabet_confirm_buffer.append(None)

        if (len(self.alphabet_confirm_buffer) == self.alphabet_confirm_buffer.maxlen and
                len(set(self.alphabet_confirm_buffer)) == 1 and
                self.alphabet_confirm_buffer[0] is not None):
            
            current_stable_prediction = self.alphabet_confirm_buffer[0]
            if current_stable_prediction != self.last_confirmed_alphabet:
                self.last_confirmed_alphabet = current_stable_prediction
                return current_stable_prediction, confidence
        
        # 연속된 글자 인식을 위해, 글자가 확정되지 않으면 마지막 확정 글자 초기화
        if self.alphabet_confirm_buffer[0] is None:
             self.last_confirmed_alphabet = None

        return None, 0.0
    
    def reset_word_buffer(self):
        """단어 예측 버퍼와 이력을 초기화합니다."""
        self.word_buffer.clear()
        self.word_history.clear()

class Visualizer:
    """
    영상 프레임에 텍스트와 랜드마크를 그리는 역할을 합니다.
    (LSTM_video_OOP2.py와 동일한 클래스)
    """
    def __init__(self, font_path="C:/Windows/Fonts/malgunbd.ttf"):
        try:
            self.font = ImageFont.truetype(font_path, 30)
            self.font_small = ImageFont.truetype(font_path, 20)
        except IOError:
            self.font = ImageFont.load_default()
            self.font_small = ImageFont.load_default()
            print(f"경고: {font_path}에서 글꼴을 찾을 수 없습니다. 기본 글꼴을 사용합니다.")
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands

    def draw(self, frame, prediction, confidence, sentence, buffer_status, results_pose, results_hands):
        """프레임에 모든 정보 텍스트와 랜드마크를 그립니다."""
        # 1. 랜드마크 그리기
        if results_pose and results_pose.pose_landmarks:
            self.mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        if results_hands and results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        # 2. 텍스트 그리기
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        display_text = "인식 결과: ..."
        if prediction:
            display_text = f"인식 결과: {prediction} ({confidence:.2f}%)"
        draw.text((10, 30), display_text, font=self.font, fill=(255, 255, 0)) # Yellow
        draw.text((10, 65), buffer_status, font=self.font_small, fill=(255, 165, 0)) # Orange
        draw.text((10, 95), f"문장: {' '.join(sentence)}", font=self.font, fill=(255, 255, 255)) # White
        
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

class SignLanguageRecognizerGUI:
    """
    실시간 웹캠 입력을 받아 수어 인식을 총괄하는 메인 클래스.
    """
    def __init__(self, config):
        self.config = config
        self.video_cap = cv2.VideoCapture(config['WEBCAM_INDEX'])
        if not self.video_cap.isOpened():
            raise IOError(f"오류: 웹캠을 열 수 없습니다 (인덱스: {config['WEBCAM_INDEX']})")
        
        # 초당 프레임(FPS)을 30으로 고정하여 유휴 시간 계산
        self.IDLE_TIME_THRESHOLD_FRAMES = int(config['IDLE_TIME_SECS'] * 30)
        
        self.feature_extractor = FeatureExtractor()
        self.predictor = Predictor(config)
        self.visualizer = Visualizer()
        
        self.sentence_words = []
        self.current_prediction = None
        self.current_confidence = 0.0
        self.idle_counter = 0
        
        self.HAND_HISTORY_LENGTH = 15 
        self.hand_presence_history = deque(maxlen=self.HAND_HISTORY_LENGTH)

    def run(self):
        """메인 인식 루프를 시작합니다."""
        print(f"\n--- 실시간 수어 인식을 시작합니다 ---")
        print(f"사용 장치: {self.predictor.device}")
        print("인식을 중단하려면 'q' 키를, 문장을 리셋하려면 'r' 키를 누르세요.")

        frame_process_counter = 0
        results_pose, results_hands = None, None

        while self.video_cap.isOpened():
            ret, frame = self.video_cap.read()
            if not ret:
                print("오류: 웹캠에서 프레임을 가져올 수 없습니다.")
                break
                
            frame = cv2.flip(frame, 1)
            frame_process_counter += 1

            # 2프레임마다 한 번씩 특징 추출 및 예측 수행 (처리 부하 감소)
            if frame_process_counte == 0:
                word_feats, alphabet_feats, movement, results_pose, results_hands, num_hands = self.feature_extractor.extract(frame)                    
                self.hand_presence_history.append(num_hands)
                
                # 움직임 감지 및 유휴 상태 처리
                if movement < self.config['MOVEMENT_THRESHOLD']:
                    self.idle_counter += 1
                else:
                    self.idle_counter = 0
                
                if self.idle_counter >= self.IDLE_TIME_THRESHOLD_FRAMES:
                    self.predictor.reset_word_buffer()
                    self.idle_counter = 0
                else:
                    # 지능형 모델 전환 로직
                    is_likely_word_gesture = any(h == 2 for h in self.hand_presence_history)
                    
                    predicted_alphabet, alphabet_conf = None, 0.0
                    
                    # 두 손 제스처가 아니며, 오른손 특징이 존재할 때만 지문자 예측 시도
                    if not is_likely_word_gesture and alphabet_feats is not None:
                        predicted_alphabet, alphabet_conf = self.predictor.predict_fingerspelling(alphabet_feats)
                    
                    if predicted_alphabet:
                        self.idle_counter = 0
                        self.current_prediction = predicted_alphabet
                        self.current_confidence = alphabet_conf * 100
                        
                        if not self.sentence_words or self.sentence_words[-1] != predicted_alphabet:
                            self.sentence_words.append(predicted_alphabet)
                        
                        self.predictor.reset_word_buffer() # 지문자 인식 시 단어 버퍼 초기화
                    else:
                        predicted_word, word_conf = self.predictor.predict_word(word_feats)
                        if predicted_word:
                            self.idle_counter = 0
                            self.current_prediction = predicted_word
                            self.current_confidence = word_conf * 100

                            if not self.sentence_words or self.sentence_words[-1] != predicted_word:
                                self.sentence_words.append(predicted_word)
                            
                            # 단어 인식 후 버퍼를 자동으로 비우므로 별도 리셋 불필요

            # 매 프레임마다 화면 업데이트 (부드러운 영상 출력)
            buffer_status = f"단어 버퍼: {len(self.predictor.word_buffer)}/{self.config['SEQ_LEN_WORD']} | 지문자 버퍼: {len(self.predictor.alphabet_buffer)}/{self.config['SEQ_LEN_ALPHABET']}"
            display_frame = self.visualizer.draw(frame, self.current_prediction, self.current_confidence, self.sentence_words, buffer_status, results_pose, results_hands)
            cv2.imshow('Real-time Sign Language Recognition', display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'): # 'r' 키로 문장 리셋
                self.sentence_words.clear()
                self.predictor.reset_word_buffer()
                self.predictor.last_confirmed_alphabet = None
                self.current_prediction = None
                print("문장이 리셋되었습니다.")
        
        self.cleanup()
            
    def cleanup(self):
        """리소스 정리 및 최종 결과 출력."""
        if self.sentence_words:
            print(f"\n--- 인식 종료 ---")
            print(f"최종 문장: {' '.join(self.sentence_words)}")
        
        self.feature_extractor.close()
        self.video_cap.release()
        cv2.destroyAllWindows()
        print("--- 리소스가 모두 해제되었습니다 ---")


if __name__ == '__main__':
    # --- 사용자 설정 ---
    CONFIG = {
        "WEBCAM_INDEX": 'http://10.101.171.170:4747/video',  # 기본 웹캠은 0, 다른 카메라는 1, 2, ...
        "MODEL_DIR": "C:/Users/bit/Desktop", # 모델 파일들이 있는 폴더 경로
        
        # 단어 모델 파라미터
        "SEQ_LEN_WORD": 45,
        "OVERLAP_LEN_WORD": 10,
        "CONF_THRESHOLD_WORD": 0.89,
        "STABLE_THRESHOLD_WORD": 1,
        
        # 지문자 모델 파라미터
        "SEQ_LEN_ALPHABET": 10,
        "CONF_THRESHOLD_ALPHABET": 0.9, # 지문자 후보가 되기 위한 최소 신뢰도

        # 일반 파라미터
        "IDLE_TIME_SECS": 2.0,       # 2초간 움직임 없으면 리셋
        "MOVEMENT_THRESHOLD": 1.0,   # 움직임 감지 임계값
    }

    try:
        recognizer = SignLanguageRecognizerGUI(CONFIG)
        recognizer.run()
    except (IOError, FileNotFoundError) as e:
        print(f"오류: {e}")
        print("MODEL_DIR 경로에 모델 파일들이 모두 있는지 확인해주세요.")
    except Exception as e:
        print(f"예상치 못한 오류가 발생했습니다: {e}")