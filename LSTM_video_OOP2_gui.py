# LSTM_video_OOP_Showcase.py

from collections import deque
import cv2
import mediapipe as mp
import numpy as np
import torch
import tensorflow as tf
import json
import os
from PIL import ImageFont, ImageDraw, Image
from gtts import gTTS
from playsound import playsound
import threading

# ------------------------------------------------------------------------------------
# 1. 기존 클래스 (FeatureExtractor, Predictor) - 변경 없음
# ------------------------------------------------------------------------------------

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
        
        num_hands_detected = 0
        if results_hands.multi_hand_landmarks:
            num_hands_detected = len(results_hands.multi_hand_landmarks)
        
        pose_features = np.zeros(16)
        
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
        self.word_buffer = deque()
        self.word_history = []
        self.alphabet_buffer = deque(maxlen=config['SEQ_LEN_ALPHABET'])
        self.alphabet_confirm_buffer = deque(maxlen=5) 
        self.last_confirmed_alphabet = None
        self.config = config

    def predict_word(self, features):
        self.word_buffer.append(features)
        if len(self.word_buffer) > self.config['SEQ_LEN_WORD']:
            self.word_buffer.popleft()
        if len(self.word_buffer) < self.config['SEQ_LEN_WORD']:
            return None, 0.0

        sequence = np.array(list(self.word_buffer))
        normalized_sequence = (sequence - self.data_mean) / (self.data_std + 1e-8)
        input_tensor = torch.tensor(normalized_sequence, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probabilities = torch.softmax(self.word_model(input_tensor), dim=1)
        
        confidence, idx = torch.max(probabilities, 1)
        confidence_item = confidence.item()

        if confidence_item < self.config['CONF_THRESHOLD_WORD']:
            return None, 0.0
        
        label = self.word_labels_map.get(idx.item(), "Unknown")
        self.word_history.append(label)

        if len(self.word_history) > self.config['STABLE_THRESHOLD_WORD']:
            self.word_history.pop(0)

        if (len(self.word_history) == self.config['STABLE_THRESHOLD_WORD'] and
                len(set(self.word_history)) == 1):
            return self.word_history[0], confidence_item
        
        return None, 0.0

    def predict_fingerspelling(self, features):
        if features is None:
            self.alphabet_confirm_buffer.append(None)
            self.last_confirmed_alphabet = None 
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
            self.last_confirmed_alphabet = None

        if (len(self.alphabet_confirm_buffer) == self.alphabet_confirm_buffer.maxlen and
                len(set(self.alphabet_confirm_buffer)) == 1 and
                self.alphabet_confirm_buffer[0] is not None):

            current_stable_prediction = self.alphabet_confirm_buffer[0]
            if current_stable_prediction != self.last_confirmed_alphabet:
                self.last_confirmed_alphabet = current_stable_prediction
                return current_stable_prediction, confidence
        return None, 0.0
    
    # ✨ 빠져 있던 이 함수를 다시 추가하는 것이 중요합니다.
    def reset_word_buffer(self):
        self.word_buffer.clear()
        self.word_history.clear()

# ------------------------------------------------------------------------------------
# 2. 새로운 Visualizer_Showcase 클래스
# ------------------------------------------------------------------------------------

class Visualizer_Showcase:
    """쇼케이스에 특화된 시각화 및 피드백 클래스"""
    def __init__(self, font_path="C:/Windows/Fonts/malgunbd.ttf"):
        try:
            self.font_title = ImageFont.truetype(font_path, 22)
            self.font_main = ImageFont.truetype(font_path, 40)
            self.font_sentence = ImageFont.truetype(font_path, 30)
            self.font_status = ImageFont.truetype(font_path, 18)
        except IOError:
            self.font_title = self.font_main = self.font_sentence = self.font_status = ImageFont.load_default()
            print(f"경고: {font_path}에서 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.last_spoken_word = ""

    def speak(self, text):
        """텍스트를 음성으로 변환하여 비동기로 재생 (UI 블로킹 방지)"""
        if not text:
            return
        
        def tts_thread():
            try:
                # 동일한 단어 반복 방지 (옵션)
                if text == self.last_spoken_word:
                    return
                self.last_spoken_word = text
                
                tts_file = "temp_speech.mp3"
                tts = gTTS(text=text, lang='ko')
                tts.save(tts_file)
                playsound(tts_file)
                os.remove(tts_file)
            except Exception as e:
                print(f"TTS 오류: {e}")
                self.last_spoken_word = "" # 오류 발생 시 초기화

        threading.Thread(target=tts_thread, daemon=True).start()

    def draw(self, frame, prediction, confidence, sentence, buffer_status, 
             results_pose, results_hands, model_status, word_appended):
        """모든 시각적 요소를 프레임에 그립니다."""
        # 랜드마크 그리기
        if results_pose and results_pose.pose_landmarks:
            self.mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                                           self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                           self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
        if results_hands and results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                                               self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                               self.mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
        
        h, w, _ = frame.shape
        
        # 하단에 반투명 대시보드 배경 추가
        panel_height = 200
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - panel_height), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # PIL을 사용하여 대시보드에 텍스트 및 그래픽 그리기
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        # 현재 인식 단어 및 신뢰도 바
        draw.text((20, h - panel_height + 10), "■ 현재 인식", font=self.font_title, fill=(255, 255, 0))
        display_text = "..."
        if prediction:
            display_text = f"{prediction}"
        
        text_color = (0, 255, 0) if word_appended else (255, 255, 255)
        draw.text((30, h - panel_height + 40), display_text, font=self.font_main, fill=text_color)
        
        bar_x, bar_y, bar_max_width, bar_height = 300, h - panel_height + 60, 200, 25
        bar_width = int((confidence / 100) * bar_max_width)
        draw.rectangle([bar_x, bar_y, bar_x + bar_max_width, bar_y + bar_height], outline=(255,255,255), width=1)
        draw.rectangle([bar_x, bar_y, bar_x + bar_width, bar_y + bar_height], fill=(0, 255, 0))
        draw.text((bar_x + bar_max_width + 10, bar_y), f"{confidence:.1f}%", font=self.font_title, fill=(0, 255, 0))

        # 누적 문장 표시
        draw.text((20, h - panel_height + 105), "■ 번역 문장", font=self.font_title, fill=(255, 255, 0))
        sentence_text = ' '.join(sentence) if sentence else "수어를 시작하세요"
        draw.text((30, h - panel_height + 135), sentence_text, font=self.font_sentence, fill=(255, 255, 255))
        
        # 시스템 상태 표시
        status_text_x = w - 250
        draw.text((status_text_x, h - panel_height + 15), "■ 시스템 상태", font=self.font_title, fill=(255, 255, 0))
        model_status_color = (137, 207, 240) # 하늘색
        draw.text((status_text_x + 10, h - panel_height + 50), f"모델: {model_status}", font=self.font_status, fill=model_status_color)
        draw.text((status_text_x + 10, h - panel_height + 80), f"버퍼: {buffer_status}", font=self.font_status, fill=(200, 200, 200))
        
        # TTS 호출
        if word_appended and prediction:
            self.speak(prediction)

        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# ------------------------------------------------------------------------------------
# 3. 최종 수정된 SignLanguageRecognizer 클래스
# ------------------------------------------------------------------------------------

class SignLanguageRecognizer:
    """
    Main class to orchestrate the sign language recognition process from a video file.
    """
    def __init__(self, config):
        self.config = config
        self.video_cap = cv2.VideoCapture(config['VIDEO_FILE_PATH'])
        if not self.video_cap.isOpened():
            raise IOError(f"오류: 비디오 파일을 열 수 없습니다: {config['VIDEO_FILE_PATH']}")
        
        fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        fps = 30 if fps == 0 else fps
        self.IDLE_TIME_THRESHOLD_FRAMES = int(config['IDLE_TIME_SECS'] * fps)
        
        # 컴포넌트 초기화
        self.feature_extractor = FeatureExtractor()
        self.predictor = Predictor(config)
        self.visualizer = Visualizer_Showcase() # 쇼케이스용 Visualizer로 교체
        
        # 상태 변수
        self.sentence_words = []
        self.current_prediction = None
        self.current_confidence = 0.0
        self.idle_counter = 0
        
        self.HAND_HISTORY_LENGTH = 15 
        self.hand_presence_history = deque(maxlen=self.HAND_HISTORY_LENGTH)

        # 쇼케이스용 상태 변수 추가
        self.model_status = "대기 중"
        self.word_appended_flag = False

    def run(self):
        """메인 인식 루프 (상세 디버깅 모드)"""
        print(f"\n--- 수어 인식 시스템 시작 (디버그 모드) ---")
        print(f"사용 디바이스: {self.predictor.device}")

        frame_process_counter = 0
        results_pose, results_hands = None, None

        while self.video_cap.isOpened():
            ret, frame = self.video_cap.read()
            if not ret:
                print("비디오 파일의 끝에 도달했습니다.")
                break
            
            frame = cv2.flip(frame, 1)
            frame_process_counter += 1
            self.word_appended_flag = False

            if frame_process_counter % 2 == 0:
                word_feats, alphabet_feats, movement, results_pose, results_hands, num_hands = self.feature_extractor.extract(frame)
                
                self.hand_presence_history.append(num_hands)

                if movement < self.config['MOVEMENT_THRESHOLD']:
                    self.idle_counter += 1
                else:
                    self.idle_counter = 0
                
                if self.idle_counter >= self.IDLE_TIME_THRESHOLD_FRAMES:
                    if self.sentence_words:
                        print(f"비활성으로 인한 문장 초기화: {' '.join(self.sentence_words)}")
                    self.sentence_words.clear()
                    self.predictor.reset_word_buffer()
                    self.idle_counter = 0
                    self.visualizer.last_spoken_word = ""

                predicted_word, word_conf = self.predictor.predict_word(word_feats)
                predicted_alphabet, alphabet_conf = self.predictor.predict_fingerspelling(alphabet_feats)

                # --- ✨ 상세 디버깅 출력 코드 ✨ ---
                # 예측 결과가 있을 때만 상세 로그를 출력합니다.
                if predicted_word or predicted_alphabet:
                    print(f"--- FRAME {frame_process_counter} DEBUG ---")
                    print(f"  Word Model   : '{predicted_word}' (Conf: {word_conf:.4f})")
                    print(f"  Alphabet Model : '{predicted_alphabet}' (Conf: {alphabet_conf:.4f})")
                    print(f"--------------------------")
                # --- ✨ 디버깅 코드 끝 ✨ ---

                if predicted_alphabet and alphabet_conf > self.config['CONF_THRESHOLD_ALPHABET'] and alphabet_conf > word_conf + 0.1:
                    self.model_status = "지문자 우세"
                    self.idle_counter = 0
                    self.current_prediction = predicted_alphabet
                    self.current_confidence = alphabet_conf * 100
                    
                    if not self.sentence_words or self.sentence_words[-1] != predicted_alphabet:
                        self.sentence_words.append(predicted_alphabet)
                        self.word_appended_flag = True
                        # print(f"지문자 추가: {predicted_alphabet} | 현재 문장: {' '.join(self.sentence_words)}")
                    
                    self.predictor.reset_word_buffer()

                elif predicted_word and word_conf > self.config['CONF_THRESHOLD_WORD']:
                    self.model_status = "단어 우세"
                    self.idle_counter = 0
                    self.current_prediction = predicted_word
                    self.current_confidence = word_conf * 100

                    if not self.sentence_words or self.sentence_words[-1] != predicted_word:
                        self.sentence_words.append(predicted_word)
                        self.word_appended_flag = True
                        # print(f"단어 추가: {predicted_word} | 현재 문장: {' '.join(self.sentence_words)}")
                        
                    self.predictor.reset_word_buffer()

            buffer_status = f"단어:{len(self.predictor.word_buffer)}/{self.config['SEQ_LEN_WORD']} | 지문자:{len(self.predictor.alphabet_buffer)}/{self.config['SEQ_LEN_ALPHABET']}"
            display_frame = self.visualizer.draw(frame, self.current_prediction, self.current_confidence, self.sentence_words, buffer_status, results_pose, results_hands, self.model_status, self.word_appended_flag)
            cv2.imshow('Sign Language Recognition Showcase', display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cleanup()

    def cleanup(self):
        """리소스 정리 및 최종 결과 출력"""
        if self.sentence_words:
            print(f"\n--- 비디오 종료 ---")
            print(f"최종 번역 문장: {' '.join(self.sentence_words)}")
        
        self.feature_extractor.close()
        self.video_cap.release()
        cv2.destroyAllWindows()
        print("--- 시스템 종료 및 리소스 해제 ---")

# ------------------------------------------------------------------------------------
# 4. 메인 실행 부분
# ------------------------------------------------------------------------------------

if __name__ == '__main__':
    # --- 사용자 설정 ---
    CONFIG = {
        "VIDEO_FILE_PATH": r"c:\Users\bit\Desktop\server\debug_videos\aaa_20250805_143155.mp4",
        "MODEL_DIR": "C:/Users/bit/Desktop",
        
        # 단어 모델 파라미터
        "SEQ_LEN_WORD": 45,
        "STABLE_THRESHOLD_WORD": 1,
        "CONF_THRESHOLD_WORD": 0.89,
        
        # 지문자 모델 파라미터
        "SEQ_LEN_ALPHABET": 10,
        "CONF_THRESHOLD_ALPHABET": 0.80,

        # 일반 파라미터
        "IDLE_TIME_SECS": 1.8,
        "MOVEMENT_THRESHOLD": 0.6,
    }

    try:
        recognizer = SignLanguageRecognizer(CONFIG)
        recognizer.run()
    except (IOError, FileNotFoundError) as e:
        print(f"파일 오류: {e}")
    except Exception as e:
        print(f"예상치 못한 오류가 발생했습니다: {e}")