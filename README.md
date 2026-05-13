# 실시간 수어 인식 시스템 (Real-time Sign Language Recognition with LSTM)
이 프로젝트는 영상 스트림에서 사용자의 동작을 실시간으로 분석하여 수어를 텍스트로 번역하는 딥러닝 기반 파이프라인입니다. 연속적인 시계열 데이터 처리에 강점이 있는 LSTM(Long Short-Term Memory) 아키텍처를 핵심으로 하며, 실시간 환경(CPU 등)에서도 끊김 없이 동작하도록 최적화되었습니다.

<img width="854" height="480" alt="Sentence Recognition 2025-07-16 19-21-25 (online-video-cutter com)" src="https://github.com/user-attachments/assets/15e0c6de-4663-453b-aed7-ec4e4dd5bf2a" />
※2배속한 영상입니다
(나, 당신, 만나다, 감사의 수어단어를 동작한 영상입니다)


# 기능 및 특징 (Key Features)
1. 모델 아키텍처 (Advanced Model Architecture)
BiLSTM + Attention Mechanism: 기본 LSTM 모델에 양방향(Bidirectional) 처리와 어텐션 메커니즘을 추가로 적용하여, 문맥의 흐름을 더 정확히 파악하고 수어 인식 성능을 대폭 향상시켰습니다.

2. 정교한 피처 엔지니어링 (Feature Engineering)
498차원 융합 피처: 수어의 의미를 정확히 구분하기 위해 손의 모양과 같은 정적 특징(Static Features)과 손의 움직임 궤적을 나타내는 동적 특징(Dynamic Features)을 결합하여 총 498차원의 고밀도 피처를 모델의 입력으로 사용합니다.

3. 데이터 파이프라인 및 전처리 (Data Pipeline)
데이터 증강(Data Augmentation): 수어 데이터셋의 고질적인 부족 문제를 해결하기 위해, 랜드마크 좌표 기반의 자체적인 데이터 증강 로직을 구현하여 모델의 강건성(Robustness)을 확보했습니다.

벡터 정규화(Vector Normalization): 사용자와 카메라 간의 거리, 신체 크기 차이로 인한 편차를 줄이기 위해 추출된 벡터 데이터에 정규화 과정을 거칩니다.

4. 실시간 안정성 및 추가 로직 (Robustness & State Control)
예측 결과 후처리 (Post-processing): 프레임 단위의 일시적인 오인식이나 흔들림을 보정하는 필터링 로직을 적용하여 최종 출력의 안정성을 높였습니다.

유휴 상태 (Idle State) 감지: 사용자가 수어를 하지 않고 대기하는 상태를 자동으로 인식하여, 불필요한 추론을 방지하고 시스템 리소스를 절약합니다.

지문자 인식 (Fingerspelling): 등록된 수어 단어 외에 고유명사 등을 표현할 때 사용하는 지문자(알파벳/자음·모음) 인식 로직이 함께 통합되어 있습니다.

<img width="2400" height="1350" alt="image" src="https://github.com/user-attachments/assets/3c6d0833-48b2-40d2-b57c-cac3f6d8d7a5" />
<img width="2400" height="1350" alt="image" src="https://github.com/user-attachments/assets/2222a8fa-7608-46fb-901f-493171da4bdc" />
<img width="2400" height="1350" alt="image" src="https://github.com/user-attachments/assets/35a9cccb-4e10-45fb-9d79-4bfd502725b2" />
<img width="2400" height="1350" alt="image" src="https://github.com/user-attachments/assets/1d6d1549-0c2c-4319-9163-5706812cb3b3" />
<img width="2400" height="1350" alt="image" src="https://github.com/user-attachments/assets/ec808253-5f23-4487-acbe-cf35f25396cd" />
<img width="2400" height="1350" alt="image" src="https://github.com/user-attachments/assets/4c6f1097-df10-4259-a1dd-8b27ef46b1fd" />
<img width="2400" height="1350" alt="image" src="https://github.com/user-attachments/assets/15f597f4-1fbe-4e34-83d9-35d8a69f69ef" />
<img width="2400" height="1350" alt="image" src="https://github.com/user-attachments/assets/401961aa-4bef-41d5-a206-4b7a47fe63d1" />
