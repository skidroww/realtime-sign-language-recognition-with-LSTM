# LSTM_video_OOP.py

from collections import deque
import cv2
import mediapipe as mp
import numpy as np
import torch
import tensorflow as tf
import json
import os
from PIL import ImageFont, ImageDraw, Image

class FeatureExtractor:
    """
    Extracts pose and hand keypoints from video frames for sign language recognition.
    It calculates features for both word and fingerspelling models and detects movement.
    """
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.pose = self.mp_pose.Pose(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.8)
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.8)
        
        self.previous_features = None
        self.previous_velocity = None

    def _get_75d_hand_features(self, joint):
        """Helper function to generate 75-dimensional hand features for the word model."""
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
        """Helper function to generate 55-dimensional hand features for the fingerspelling model."""
        v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :2]
        v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :2]
        v = v2 - v1
        v = v / (np.linalg.norm(v, axis=1)[:, np.newaxis] + 1e-6)
        angle = np.degrees(np.arccos(np.einsum('nt,nt->n',
            v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
            v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])))
        return np.concatenate([v.flatten(), angle])

    def extract(self, frame):
        """Processes a single frame to extract all required features."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_pose = self.pose.process(frame_rgb)
        results_hands = self.hands.process(frame_rgb)
        
        # --- 1. Word Model Feature Extraction (Pose + Hands) ---
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
                
                # Feature for word model
                hand_features_75d = self._get_75d_hand_features(joint)
                if handedness == "Left":
                    left_hand_features = hand_features_75d
                elif handedness == "Right":
                    right_hand_features = hand_features_75d
                    # Feature for alphabet model (right hand only)
                    alphabet_feature = self._get_55d_hand_features(joint)

        word_model_features = np.concatenate([pose_features, left_hand_features, right_hand_features])

        # --- 2. Dynamic Feature Calculation (Velocity, Acceleration) ---
        if self.previous_features is None: self.previous_features = np.zeros_like(word_model_features)
        velocity = word_model_features - self.previous_features
        if self.previous_velocity is None: self.previous_velocity = np.zeros_like(velocity)
        acceleration = velocity - self.previous_velocity
        
        movement = np.sum(np.abs(velocity))
        self.previous_features = word_model_features
        self.previous_velocity = velocity

        dynamic_word_features = np.concatenate([word_model_features, velocity, acceleration])
        
        return dynamic_word_features, alphabet_feature, movement

    def close(self):
        """Releases Mediapipe resources."""
        self.pose.close()
        self.hands.close()


class Predictor:
    """
    Manages loading and running both the word (PyTorch) and 
    fingerspelling (TFLite) models.
    """
    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # --- Word Model (PyTorch) ---
        self.word_model = torch.jit.load(os.path.join(config['MODEL_DIR'], "lstm_sign_language_model_scripted.pt")).to(self.device)
        self.word_model.eval()
        self.data_mean = np.load(os.path.join(config['MODEL_DIR'], "data_mean.npy"))
        self.data_std = np.load(os.path.join(config['MODEL_DIR'], "data_std.npy"))
        with open(os.path.join(config['MODEL_DIR'], "label_map.json"), 'r', encoding='utf-8') as f:
            self.word_labels_map = {v: k for k, v in json.load(f).items()}
        
        # --- Fingerspelling Model (TFLite) ---
        tflite_path = os.path.join(config['MODEL_DIR'], "multi_hand_gesture_classifier.tflite")
        self.alphabet_interpreter = tf.lite.Interpreter(model_path=tflite_path)
        self.alphabet_interpreter.allocate_tensors()
        self.alphabet_input_details = self.alphabet_interpreter.get_input_details()
        self.alphabet_output_details = self.alphabet_interpreter.get_output_details()
        self.alphabet_actions = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ','ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ','ㅐ', 'ㅒ', 'ㅔ', 'ㅖ', 'ㅢ', 'ㅚ', 'ㅟ']

        # --- Buffers and Parameters ---
        #self.word_buffer = deque(maxlen=config['SEQ_LEN_WORD'])
        self.word_buffer = deque()
        self.word_history = []
        self.alphabet_buffer = deque(maxlen=config['SEQ_LEN_ALPHABET'])
        
        self.alphabet_confirm_buffer = deque(maxlen=3) # 최근 3개 예측 저장용 버퍼
        self.last_confirmed_alphabet = None # 마지막으로 인식된 지문자를 저장
        
        self.config = config
        

    def predict_word(self, features):
        """
        특징(features) 시퀀스로부터 단어를 예측합니다.
        '호핑 윈도우' 방식을 사용하여 효율적으로 버퍼를 관리합니다.
        """
        # 1. 버퍼에 새로운 특징 추가
        self.word_buffer.append(features)

        # 2. 버퍼가 예측에 필요한 최소 길이에 도달했는지 확인
        if len(self.word_buffer) < self.config['SEQ_LEN_WORD']:
            return None, 0.0

        # 3. 예측 수행
        # 현재 버퍼의 앞부분(SEQ_LEN_WORD 만큼)을 numpy 배열로 변환
        sequence = np.array(list(self.word_buffer)[:self.config['SEQ_LEN_WORD']])
        
        # 데이터 정규화 및 PyTorch 텐서로 변환
        normalized_sequence = (sequence - self.data_mean) / (self.data_std + 1e-8)
        input_tensor = torch.tensor(normalized_sequence, dtype=torch.float32).unsqueeze(0).to(self.device)

        # 모델 추론 실행
        with torch.no_grad():
            probabilities = torch.softmax(self.word_model(input_tensor), dim=1)
        
        # 가장 높은 확률의 예측값과 신뢰도 추출
        confidence, idx = torch.max(probabilities, 1)
        confidence_item = confidence.item()

        # 4. 버퍼 업데이트 (호핑 윈도우 구현)
        # 예측에 사용된 분량에서 오버랩(overlap) 길이를 제외한 만큼 버퍼의 앞에서 제거
        # 이를 통해 윈도우가 한 칸씩 미끄러지는 것이 아니라, 정해진 'hop_size'만큼 건너뛰게 됨
        hop_size = self.config['SEQ_LEN_WORD'] - self.config['OVERLAP_LEN_WORD']
        for _ in range(hop_size):
            self.word_buffer.popleft() # deque의 popleft()는 O(1)로 매우 효율적

        # 5. 예측 안정화 및 결과 반환
        # 신뢰도가 임계값 미만이면 무시
        if confidence_item < self.config['CONF_THRESHOLD_WORD']:
            return None, 0.0
        
        # 예측된 레이블 확인
        label = self.word_labels_map.get(idx.item(), "Unknown")
        self.word_history.append(label)

        # word_history 버퍼는 STABLE_THRESHOLD_WORD 만큼의 예측 기록을 유지
        if len(self.word_history) > self.config['STABLE_THRESHOLD_WORD']:
            self.word_history.pop(0)

        # 최근 예측들이 모두 동일해야 안정된 예측으로 간주하고 최종 결과 반환
        if (len(self.word_history) == self.config['STABLE_THRESHOLD_WORD'] and
                len(set(self.word_history)) == 1):
            return self.word_history[0], confidence_item
        
        # 안정화 조건을 만족하지 못하면 결과 없음으로 처리
        return None, 0.0

    def predict_fingerspelling(self, features):
        """
        특징 시퀀스로부터 지문자를 예측합니다. (최종 수정 버전)
        Debouncing과 마지막 인식 글자 비교를 통해 안정성을 대폭 향상시킵니다.
        """
        # --- 1. 특징을 버퍼에 추가 ---
        if features is None:
            self.alphabet_confirm_buffer.append(None)
            # 손이 감지되지 않으면, 마지막 인식 기록을 리셋하여 다음 글자를 바로 인식하게 함
            self.last_confirmed_alphabet = None 
            return None, 0.0

        self.alphabet_buffer.append(features)
        if len(self.alphabet_buffer) < self.config['SEQ_LEN_ALPHABET']:
            return None, 0.0

        # --- 2. TFLite 모델로 예측 수행 ---
        input_data = np.expand_dims(np.array(self.alphabet_buffer, dtype=np.float32), axis=0)
        self.alphabet_interpreter.set_tensor(self.alphabet_input_details[0]['index'], input_data)
        self.alphabet_interpreter.invoke()
        y_pred = self.alphabet_interpreter.get_tensor(self.alphabet_output_details[0]['index'])

        # deque의 maxlen 기능에 의해 가장 오래된 데이터는 자동으로 삭제됨
        # del self.alphabet_buffer[0] # 이 줄은 필요 없거나 로직에 따라 유지

        i_pred = int(np.argmax(y_pred[0]))
        confidence = y_pred[0][i_pred]
        
        predicted_char_for_debug = self.alphabet_actions[i_pred]
        print(f"Debug: Char='{predicted_char_for_debug}', Conf={confidence:.4f}, Buffer={list(self.alphabet_confirm_buffer)}")

        # --- 3. 확정용 버퍼에 예측 결과 추가 ---
        if confidence > self.config['CONF_THRESHOLD_ALPHABET']:
            self.alphabet_confirm_buffer.append(self.alphabet_actions[i_pred])
        else:
            self.alphabet_confirm_buffer.append(None)
            # 신뢰도 미달 시에도 마지막 인식 기록을 리셋
            self.last_confirmed_alphabet = None

        # --- 4. 핵심 수정: 버퍼를 clear()하는 대신, '마지막으로 확정된 글자'와 비교 ---
        if (len(self.alphabet_confirm_buffer) == 3 and
                len(set(self.alphabet_confirm_buffer)) == 1 and
                self.alphabet_confirm_buffer[0] is not None):

            current_stable_prediction = self.alphabet_confirm_buffer[0]

            # 현재 안정된 예측이 '마지막으로 성공한 예측'과 다를 때만 결과 반환
            if current_stable_prediction != self.last_confirmed_alphabet:
                self.last_confirmed_alphabet = current_stable_prediction # 마지막 성공 예측을 업데이트
                return current_stable_prediction, confidence

        # 모든 조건이 만족되지 않으면 결과 없음으로 처리
        return None, 0.0
    
    def reset_word_buffer(self):
        """Resets the word prediction buffer and history."""
        self.word_buffer.clear()
        self.word_history.clear()


class Visualizer:
    """Handles drawing text and results onto the video frame."""
    def __init__(self, font_path="C:/Windows/Fonts/malgunbd.ttf"):
        try:
            self.font = ImageFont.truetype(font_path, 30)
            self.font_small = ImageFont.truetype(font_path, 20)
        except IOError:
            self.font = ImageFont.load_default()
            self.font_small = ImageFont.load_default()
            print(f"Warning: Font not found at {font_path}. Using default font.")

    def draw(self, frame, prediction, confidence, sentence, buffer_status):
        """Draws all informational text on the frame."""
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # Display current prediction
        display_text = "인식 결과: ..."
        if prediction:
            display_text = f"인식 결과: {prediction} ({confidence:.2f}%)"
        draw.text((10, 30), display_text, font=self.font, fill=(255, 255, 0)) # Yellow
        
        # Display buffer status
        draw.text((10, 65), buffer_status, font=self.font_small, fill=(255, 165, 0)) # Orange
        
        # Display the accumulating sentence
        draw.text((10, 95), f"문장: {' '.join(sentence)}", font=self.font, fill=(255, 255, 255)) # White
        
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


class SignLanguageRecognizer:
    """
    Main class to orchestrate the sign language recognition process from a video file.
    """
    def __init__(self, config):
        self.config = config
        self.video_cap = cv2.VideoCapture(config['VIDEO_FILE_PATH'])
        if not self.video_cap.isOpened():
            raise IOError(f"Error: Could not open video file: {config['VIDEO_FILE_PATH']}")
        
        fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        fps = 30 if fps == 0 else fps
        self.IDLE_TIME_THRESHOLD_FRAMES = int(config['IDLE_TIME_SECS'] * fps)
        
        # Initialize components
        self.feature_extractor = FeatureExtractor()
        self.predictor = Predictor(config)
        self.visualizer = Visualizer()
        
        # State variables
        self.sentence_words = []
        self.current_prediction = None
        self.current_confidence = 0.0
        self.idle_counter = 0
        
        self.word_cooldown_counter = 0 # 단어 예측 쿨다운 타이머


    def run(self):
        """Starts the main recognition loop."""
        print(f"\n--- Starting Sign Language Recognition ---")
        print(f"Device: {self.predictor.device}")

        frame_process_counter = 0

        ret, frame = self.video_cap.read()
        if not ret:
            print("오류: 비디오 파일을 읽을 수 없습니다. 경로를 확인해 주세요.")
            self.cleanup()
            return
        display_frame = cv2.flip(frame, 1)

        while self.video_cap.isOpened():
            ret, frame = self.video_cap.read()
            if not ret:
                print("End of video file reached.")
                break
            
            frame = cv2.flip(frame, 1)
            frame_process_counter += 1

            if frame_process_counter % 2 == 0:
                # --- 쿨다운 로직: 카운터가 0보다 크면 1씩 감소 ---
                if self.word_cooldown_counter > 0:
                    self.word_cooldown_counter -= 1
                
                # --- 1. 특징 추출 ---
                word_feats, alphabet_feats, movement = self.feature_extractor.extract(frame)

                # --- 2. 휴지 상태 감지 (이전과 동일) ---
                if movement < self.config['MOVEMENT_THRESHOLD']:
                    self.idle_counter += 1
                else:
                    self.idle_counter = 0
                
                if self.idle_counter >= self.IDLE_TIME_THRESHOLD_FRAMES:
                    if self.sentence_words:
                        print(f"Final Sentence (due to inactivity): {' '.join(self.sentence_words)}")
                    self.sentence_words.clear()
                    self.predictor.reset_word_buffer()
                    self.idle_counter = 0

                # --- 3. 예측 로직 (구조가 매우 중요합니다) ---
                predicted_alphabet, alphabet_conf = self.predictor.predict_fingerspelling(alphabet_feats)
                
                # ▼ [수정됨] 3-1. 지문자가 인식된 경우
                if predicted_alphabet:
                    self.idle_counter = 0
                    self.current_prediction = predicted_alphabet
                    self.current_confidence = alphabet_conf * 100
                    
                    # 중복 추가 방지
                    if not self.sentence_words or self.sentence_words[-1] != predicted_alphabet:
                        self.sentence_words.append(predicted_alphabet)
                        # "Fingerspelling Appended"로 올바르게 출력
                        print(f"Fingerspelling Appended: {predicted_alphabet} | Current Sentence: {' '.join(self.sentence_words)}")
                    
                    # 지문자가 인식되면 단어 버퍼와 쿨다운을 모두 초기화
                    self.predictor.reset_word_buffer()
                    self.word_cooldown_counter = 0

                # ▼ [수정됨] 3-2. 지문자가 없고, 쿨다운 상태가 아닐 때만 단어 예측 시도
                elif self.word_cooldown_counter == 0:
                    predicted_word, word_conf = self.predictor.predict_word(word_feats)
                    
                    if predicted_word:
                        self.idle_counter = 0
                        self.current_prediction = predicted_word
                        self.current_confidence = word_conf * 100

                        # 중복 추가 방지
                        if not self.sentence_words or self.sentence_words[-1] != predicted_word:
                            self.sentence_words.append(predicted_word)
                            # "Word Appended"로 올바르게 출력
                            print(f"Word Appended: {predicted_word} | Current Sentence: {' '.join(self.sentence_words)}")
                            
                            # ★★★ 단어가 성공적으로 추가되었을 때만 쿨다운을 설정합니다 ★★★
                            self.word_cooldown_counter = 15 
                
                # ▼ [수정됨] 3-3. 지문자가 없고, 쿨다운 상태일 때는 아무것도 하지 않음
                else:
                    pass # 쿨다운 중이므로 단어 예측을 건너뜁니다.

                # --- 4. 시각화 (이전과 동일) ---
                buffer_status = f"Word Buf: {len(self.predictor.word_buffer)}/{self.config['SEQ_LEN_WORD']} | Alpha Buf: {len(self.predictor.alphabet_buffer)}/{self.config['SEQ_LEN_ALPHABET']}"
                display_frame = self.visualizer.draw(frame, self.current_prediction, self.current_confidence, self.sentence_words, buffer_status)

            cv2.imshow('Sign Language Recognition', display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        self.cleanup()
            
    def cleanup(self):
        """Cleans up resources and prints the final result."""
        if self.sentence_words:
            print(f"\n--- Video Finished ---")
            print(f"Final Sentence: {' '.join(self.sentence_words)}")
        
        self.feature_extractor.close()
        self.video_cap.release()
        cv2.destroyAllWindows()
        print("--- Resources Released ---")


if __name__ == '__main__':
    # --- User Configuration ---
    CONFIG = {
        "VIDEO_FILE_PATH": r"c:\Users\bit\Desktop\video\KakaoTalk_20250715_210919628.mp4",
        "MODEL_DIR": "C:/Users/bit/Desktop",
        
        # Word Model Parameters
        "SEQ_LEN_WORD": 60,
        "OVERLAP_LEN_WORD": 40,
        "STABLE_THRESHOLD_WORD": 1,
        "CONF_THRESHOLD_WORD": 0.89,
        
        # Fingerspelling Model Parameters
        "SEQ_LEN_ALPHABET": 10,
        "CONF_THRESHOLD_ALPHABET": 0.80,

        # General Parametersss
        "IDLE_TIME_SECS": 2.5,
        "MOVEMENT_THRESHOLD": 0.5,
    }

    try:
        recognizer = SignLanguageRecognizer(CONFIG)
        recognizer.run()
    except (IOError, FileNotFoundError) as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")