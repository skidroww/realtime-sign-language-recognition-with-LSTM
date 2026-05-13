import os
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from torch.optim.lr_scheduler import StepLR
import json
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau


class SignLanguageDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, seq_len=90, augment=False, num_augmentations=49, 
                 noise_strength=0.01, scale_range=(0.9, 1.1), h_flip_prob=0.0,  # <-- 1. 반전 확률 인자 추가
                 mean=None, std=None):
        self.data = data
        self.labels = labels
        self.seq_len = seq_len
        self.augment = augment
        self.num_augmentations_per_sample = num_augmentations
        self.total_augmentations = num_augmentations + 1
        self.noise_strength = noise_strength
        self.scale_range = scale_range
        self.h_flip_prob = h_flip_prob  # <-- 2. 인자 저장
        self.mean = mean
        self.std = std

    def __len__(self):
        if self.augment:
            return len(self.data) * self.total_augmentations
        else:
            return len(self.data)

    def __getitem__(self, idx):
        # 어떤 원본 데이터를 기반으로 할지 결정
        original_idx = idx // self.total_augmentations if self.augment else idx
        
        sequence = self.data[original_idx].copy()
        label = self.labels[original_idx]

        # 증강이 필요한 경우 (훈련 데이터셋용)
        # idx % self.total_augmentations가 0이 아니면 증강된 샘플임
        is_augmented_sample = (idx % self.total_augmentations != 0)
        if self.augment and is_augmented_sample:
            noise = np.random.normal(0, self.noise_strength, sequence.shape)
            sequence += noise
            scale_factor = np.random.uniform(self.scale_range[0], self.scale_range[1])
            sequence *= scale_factor
            
            # ---  3. 좌우 반전(Horizontal Flip) 증강 로직 추가 ---
            if np.random.rand() < self.h_flip_prob:
                # 특징 벡터 구조: [포즈(16) + 왼손(75) + 오른손(75)] -> static_dim = 166
                # 전체 구조: [static(166), velocity(166), acceleration(166)]
                static_dim = 166
                
                for i in range(3): # 위치, 속도, 가속도에 대해 모두 적용
                    offset = i * static_dim
                    
                    # 3-1. 포즈의 x좌표 반전 (x, y, x, y, ...)
                    sequence[:, offset:offset+16:2] *= -1

                    # 3-2. 손의 x좌표 반전
                    # 왼손 (v(60) + angle(15))
                    # v벡터는 [x,y,z, x,y,z, ...] 이므로 3칸마다 x좌표
                    sequence[:, offset+16:offset+16+60:3] *= -1 
                    # 오른손
                    sequence[:, offset+16+75:offset+16+75+60:3] *= -1
                    
                    # 3-3. 왼손-오른손 특징 블록 교체
                    left_hand_start = offset + 16
                    left_hand_end = offset + 16 + 75
                    right_hand_start = offset + 16 + 75
                    right_hand_end = offset + 16 + 75 + 75
                    
                    # copy()를 사용해 원본을 보존하며 교체
                    left_hand_block = sequence[:, left_hand_start:left_hand_end].copy()
                    right_hand_block = sequence[:, right_hand_start:right_hand_end].copy()
                    
                    sequence[:, left_hand_start:left_hand_end] = right_hand_block
                    sequence[:, right_hand_start:right_hand_end] = left_hand_block

        
        # 패딩 또는 잘라내기
        if sequence.shape[0] < self.seq_len:
            padding = np.zeros((self.seq_len - sequence.shape[0], sequence.shape[1]))
            sequence = np.vstack([sequence, padding])
        elif sequence.shape[0] > self.seq_len:
            sequence = sequence[:self.seq_len]

        # 표준화
        if self.mean is not None and self.std is not None:
            sequence = (sequence - self.mean) / (self.std + 1e-8)
            
        return torch.tensor(sequence).float(), torch.tensor(label).long()

# --- EarlyStopping 클래스 추가 ---
class EarlyStopping:
    """
    검증 손실이 개선되지 않을 때 학습을 조기에 중단시키는 클래스.
    """
    def __init__(self, patience=10, min_delta=0.0001, verbose=True, mode='min'):
        """
        Args:
            patience (int): 개선이 없을 때 기다릴 에폭 수.
            min_delta (float): 개선으로 간주할 최소 변화량.
            verbose (bool): 조기 중단 메시지 출력 여부.
            mode (str): 'min'이면 손실 감소, 'max'이면 정확도 증가를 모니터링.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.mode = mode
        self.best_score = None
        self.epochs_no_improve = 0
        self.early_stop = False
        self.best_model_state = None # 최적 모델의 상태 저장

        if self.mode == 'min':
            self.best_score = float('inf')
        else: # mode == 'max'
            self.best_score = float('-inf')

    def __call__(self, current_score, model):
        """
        현재 점수를 기반으로 조기 중단 여부를 확인하고, 최적 모델 상태를 저장합니다.
        Args:
            current_score (float): 현재 에폭의 모니터링 점수 (예: 검증 손실).
            model (torch.nn.Module): 현재 모델 객체.
        Returns:
            bool: 조기 중단해야 하면 True, 아니면 False.
        """
        if self.mode == 'min':
            if current_score < self.best_score - self.min_delta:
                self.best_score = current_score
                self.epochs_no_improve = 0
                self.best_model_state = model.state_dict() # 최적 모델 상태 저장
                if self.verbose:
                    print(f"Validation score improved ({self.best_score:.4f}). Saving best model state.")
            else:
                self.epochs_no_improve += 1
                if self.verbose:
                    print(f"Validation score did not improve for {self.epochs_no_improve} epochs.")
                if self.epochs_no_improve >= self.patience:
                    self.early_stop = True
                    if self.verbose:
                        print(f"Early stopping triggered after {self.patience} epochs without improvement.")
        else: # mode == 'max'
            if current_score > self.best_score + self.min_delta:
                self.best_score = current_score
                self.epochs_no_improve = 0
                self.best_model_state = model.state_dict() # 최적 모델 상태 저장
                if self.verbose:
                    print(f"Validation score improved ({self.best_score:.4f}). Saving best model state.")
            else:
                self.epochs_no_improve += 1
                if self.verbose:
                    print(f"Validation score did not improve for {self.epochs_no_improve} epochs.")
                if self.epochs_no_improve >= self.patience:
                    self.early_stop = True
                    if self.verbose:
                        print(f"Early stopping triggered after {self.patience} epochs without improvement.")
        return self.early_stop


# --- 2. LSTM 모델 정의 (이전과 동일) ---
# BiLSTM 위에 어텐션 계층을 추가한 새로운 모델
class BiLSTMAttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout_prob=0.5):
        super(BiLSTMAttentionModel, self).__init__()
        
        # 1. BiLSTM 층 (기존과 동일)
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=dropout_prob if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # 2. 어텐션 스코어를 계산하기 위한 작은 신경망 (어텐션 층)
        # BiLSTM의 출력(hidden_dim * 2)을 받아, 각 프레임의 중요도(점수 1개)를 출력
        self.attention_layer = nn.Linear(hidden_dim * 2, 1)
        
        # 3. 최종 분류기 (Classifier)
        # 어텐션을 통해 나온 '문맥 벡터'를 입력으로 받음. 크기는 hidden_dim * 2
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        
        # BiLSTM 통과
        # lstm_out shape: (batch_size, seq_len, hidden_dim * 2)
        lstm_out, _ = self.lstm(x)
        
        # --- 어텐션 메커니즘 계산 ---
        # 1. 어텐션 스코어 계산: 각 프레임이 얼마나 중요한지 점수를 매김
        # (batch_size, seq_len, hidden_dim * 2) -> (batch_size, seq_len, 1)
        attention_scores = self.attention_layer(lstm_out)
        attention_scores = attention_scores.squeeze(2) # (batch_size, seq_len)
        
        # 2. 어텐션 가중치 계산: 점수를 소프트맥스 함수로 정규화하여 총합이 1인 확률로 변환
        attention_weights = F.softmax(attention_scores, dim=1) # (batch_size, seq_len)
        
        # 3. 문맥 벡터(Context Vector) 생성: 계산된 가중치를 BiLSTM 출력에 곱하여 가중 평균을 냄
        # 중요한 프레임의 정보가 더 많이 반영된 '핵심 요약' 벡터를 생성
        # (batch_size, 1, seq_len) x (batch_size, seq_len, hidden_dim*2) -> (batch_size, 1, hidden_dim*2)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), lstm_out)
        context_vector = context_vector.squeeze(1) # (batch_size, hidden_dim*2)
        # --- 어텐션 계산 끝 ---

        # 최종 예측: '핵심 요약'인 문맥 벡터를 분류기에 통과
        output = self.fc(context_vector)
        return output

# --- 3. 하이퍼파라미터 및 메인 실행 블록 ---
if __name__ == "__main__":
    # --- 하이퍼파라미터 및 경로 설정 ---
    BASE_DIR = 'C:\\Users\\bit\\Desktop\\sign_language_data'
    categories = ["어지럽다", "나", "쉬다", "어깨","감사","만나다","잘하다",
             "안녕하세요","무엇","비빔밥","기쁘다","취미","영화","얼굴",
             "보다","이름","같다","죄송","먹다","괜찮다","수고","나이",
             "다시","얼마","날","좋다","날짜","우리","전철","버스",
             "타다","휴대전화","어디","위치","책임","도착","가족","시간",
             "소개","주세요","물음","걷다","자매",
             "공부","사람","지금","어제","겨루다","당신",
             "결혼","노력","아니다","땀","아직","결과",
             "낳다","성공","서울","저녁","고객",
             "바라다"]
    
    output_dir = "C:/Users/bit/Desktop"
    os.makedirs(output_dir, exist_ok=True)
    
    label_map = {category: i for i, category in enumerate(categories)}
    with open(os.path.join(output_dir, 'label_map.json'), 'w', encoding='utf-8') as f:
        json.dump(label_map, f, ensure_ascii=False, indent=4)
    print(f"Label mapping saved. Total {len(categories)} classes.")

    SEQ_LEN = 90
    INPUT_DIM = 498
    HIDDEN_DIM = 256
    NUM_LAYERS = 3
    DROPOUT_PROB = 0.5
    BATCH_SIZE = 64  # 메모리 오류 발생 시 이 값을 32, 16 등으로 줄여보세요.
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4 # 과적합 방지를 위한 가중치 감쇠 추가
    NUM_EPOCHS = 100
    AUGMENT_FACTOR = 19
    EARLY_STOPPING_PATIENCE = 10
    H_FLIP_PROB = 0.5  #  50% 확률로 좌우 반전 증강을 적용


    # --- 실시간 증강을 위한 데이터 처리 파이프라인 ---
    print("--- Loading Original Raw Data ---")
    all_data = []
    all_labels = []
    for category in categories:
        category_coord_path = os.path.join(BASE_DIR, category, "좌표")
        if not os.path.exists(category_coord_path):
            print(f"Warning: Path not found, skipping: {category_coord_path}")
            continue
        for file_name in os.listdir(category_coord_path):
            if file_name.lower().endswith(".npy"):
                file_path = os.path.join(category_coord_path, file_name)
                try:
                    all_data.append(np.load(file_path))
                    all_labels.append(label_map[category])
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    
    print(f"Total original samples loaded: {len(all_data)}")

    # 원본 데이터를 훈련/검증/테스트 세트로 분할
    X_train_orig, X_temp, y_train_orig, y_temp = train_test_split(
        all_data, all_labels, test_size=0.4, random_state=42, stratify=all_labels)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # 표준화 파라미터 계산 (원본 훈련 데이터 기준)
    temp_train_for_std = [seq[:SEQ_LEN] if len(seq) > SEQ_LEN else np.pad(seq, ((0, SEQ_LEN - len(seq)), (0,0)), 'constant') for seq in X_train_orig]
    data_mean = np.mean(np.array(temp_train_for_std), axis=(0, 1))
    data_std = np.std(np.array(temp_train_for_std), axis=(0, 1))
    
    np.save(os.path.join(output_dir, 'data_mean.npy'), data_mean)
    np.save(os.path.join(output_dir, 'data_std.npy'), data_std)
    print("Standardization mean and std saved.")

    # Dataset 객체 생성
    train_dataset = SignLanguageDataset(X_train_orig, y_train_orig, seq_len=SEQ_LEN, augment=True, 
                                        num_augmentations=AUGMENT_FACTOR, 
                                        h_flip_prob=H_FLIP_PROB,  
                                        mean=data_mean, std=data_std)
    val_dataset = SignLanguageDataset(X_val, y_val, seq_len=SEQ_LEN, mean=data_mean, std=data_std)
    test_dataset = SignLanguageDataset(X_test, y_test, seq_len=SEQ_LEN, mean=data_mean, std=data_std)

    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"DataLoaders created. Train items: {len(train_dataset)}, Val items: {len(val_dataset)}, Test items: {len(test_dataset)}")

    # --- 학습 준비 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train_orig), y=y_train_orig)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    model = BiLSTMAttentionModel(INPUT_DIM, HIDDEN_DIM, len(categories), NUM_LAYERS, DROPOUT_PROB).to(device)
    print(model)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True) # 스케줄러 변경
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, verbose=True)

    # --- 모델 학습 ---
    print("\n--- Starting Model Training ---")
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for val_X, val_y in val_loader:
                val_X, val_y = val_X.to(device), val_y.to(device)
                outputs = model(val_X)
                loss = criterion(outputs, val_y)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += val_y.size(0)
                val_correct += (predicted == val_y).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f}")
        
        scheduler.step(avg_val_loss)
        if early_stopping(avg_val_loss, model):
            break
            
    # --- 테스트 및 저장 ---
    print("\n--- Loading best model state for final evaluation ---")
    if early_stopping.best_model_state:
        model.load_state_dict(early_stopping.best_model_state)
    else:
        print("Warning: Early stopping did not save a best model state. Using the last model state.")


    print("\n--- Model Evaluation on Test Data ---")
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, preds = torch.max(outputs, 1)
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            
    print(classification_report(y_true, y_pred, target_names=categories, zero_division=0))
    print(confusion_matrix(y_true, y_pred))

    print("\n--- Saving Model ---")
    model.to('cpu')
    scripted_model = torch.jit.script(model)
    model_path = os.path.join(output_dir, "lstm_sign_language_model_scripted.pt")
    scripted_model.save(model_path)
    print(f"Model saved successfully to {model_path}")