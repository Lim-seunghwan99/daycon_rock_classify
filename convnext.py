# convnext.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms # torchvision.transforms 사용
import torch.nn.functional as F # FocalLoss 등에서 필요
import timm
import torchmetrics
from tqdm import tqdm
import time
import os
import numpy as np # 필요시 사용

# --- 1. Focal Loss 클래스 정의 (이전 코드에서 복사) ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2., reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            # alpha 처리 로직 (이전 코드 참조)
            if isinstance(self.alpha, (float, int)):
                 alpha = torch.tensor([self.alpha] * inputs.shape[1], device=inputs.device)
            elif isinstance(self.alpha, list):
                 alpha = torch.tensor(self.alpha, device=inputs.device, dtype=torch.float32)
            elif torch.is_tensor(self.alpha):
                 alpha = self.alpha.to(device=inputs.device, dtype=torch.float32)
            else:
                 raise TypeError("alpha must be float, list or torch.Tensor")

            if alpha.shape[0] != inputs.shape[1]:
                 raise ValueError(f"alpha size {alpha.shape[0]} does not match C {inputs.shape[1]}")

            alpha_t = alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else: # 'none'
            return focal_loss

# --- 2. 학습 및 검증 함수 정의 (이전 코드에서 복사) ---
# def train_and_validate_best_f1(...):
#     ... (이전 코드 전체 복사) ...
# --- 여기에 train_and_validate_best_f1 함수 코드를 붙여넣으세요 ---
def train_and_validate_best_f1(model: nn.Module,
                               train_loader: DataLoader,
                               val_loader: DataLoader,
                               optimizer: optim.Optimizer, # torch.optim 임포트 사용
                               criterion: nn.Module, # Loss function
                               epochs: int,
                               device: torch.device,
                               num_classes: int,
                               save_dir: str, # 디렉토리로 변경
                               model_name_base: str, # 모델 파일 이름용
                               top_k: int = 5, # 저장할 상위 모델 개수 (기본값 5)
                               gradient_clipping: float = None,
                               lr_scheduler = None,
                               warmup_epochs: int = 0,
                               base_lr: float = 1e-5
                              ):
    
    history = {'train_losses': [], 'val_losses': [], 'val_macro_f1_scores': []}
    max_val_f1 = 0.0  # 여전히 전체 최고 점수 추적 (조기 종료용)
    best_epoch = -1 # 최고 점수 달성 에폭
    top_k_checkpoints = [] # (f1_score, file_path) 튜플을 저장할 리스트

    f1_metric = torchmetrics.F1Score(task='multiclass', num_classes=num_classes, average='macro').to(device)
    patience = 5
    epochs_no_improve = 0 # 조기 종료 카운터는 여전히 max_val_f1 기준

    # save_dir 존재 확인 및 생성 (함수 호출 전에 해도 되지만 여기서도 확인)
    os.makedirs(save_dir, exist_ok=True)

    print(f"학습 시작: 총 {epochs} 에폭, Device: {device}")
    print(f"Top-{top_k} 모델 저장 디렉토리: {save_dir}") # 경로 출력 수정
    print(f"평가 기준: Validation Macro F1 Score")

    for epoch in range(epochs):
        epoch_start_time = time.time()
        current_lr = optimizer.param_groups[0]['lr']

        # --- Warmup 및 LR 스케줄링 ---
        if epoch < warmup_epochs:
            warmup_factor = (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = base_lr * warmup_factor
            current_lr = optimizer.param_groups[0]['lr']
        elif lr_scheduler is not None:
             if epoch == warmup_epochs:
                 for param_group in optimizer.param_groups:
                     param_group['lr'] = base_lr
                 lr_scheduler.step()
             else:
                 lr_scheduler.step()
             current_lr = optimizer.param_groups[0]['lr']

        # --- 학습 단계 ---
        model.train()
        running_train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train] LR: {current_lr:.1e}", leave=False)
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            if gradient_clipping is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()
            running_train_loss += loss.item()
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")
        epoch_train_loss = running_train_loss / len(train_loader)
        history['train_losses'].append(epoch_train_loss)

        # --- 검증 단계 ---
        model.eval()
        running_val_loss = 0.0
        f1_metric.reset()
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val] ", leave=False)
        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                f1_metric.update(outputs, labels)
                val_pbar.set_postfix(loss=f"{loss.item():.4f}")
        epoch_val_loss = running_val_loss / len(val_loader)
        history['val_losses'].append(epoch_val_loss)
        epoch_val_f1 = f1_metric.compute().item()
        history['val_macro_f1_scores'].append(epoch_val_f1)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        print(f"Epoch [{epoch+1}/{epochs}] ({epoch_duration:.2f}s) - "
              f"Train Loss: {epoch_train_loss:.4f}, "
              f"Val Loss: {epoch_val_loss:.4f}, "
              f"Val Macro F1: {epoch_val_f1:.4f}")

        # --- 모델 저장 및 조기 종료 ---
        is_top_k = len(top_k_checkpoints) < top_k or epoch_val_f1 > top_k_checkpoints[-1][0]

        if is_top_k:
            # 새 모델 저장 경로 생성 (에폭과 F1 점수 포함)
            checkpoint_filename = f"{model_name_base}_epoch{epoch+1}_f1_{epoch_val_f1:.4f}.pth"
            checkpoint_path = os.path.join(save_dir, checkpoint_filename)

            try:
                torch.save(model.state_dict(), checkpoint_path)
                print(f"  Checkpoint saved to {checkpoint_path}")

                # Top-K 리스트 업데이트
                top_k_checkpoints.append((epoch_val_f1, checkpoint_path))
                # F1 점수 기준 내림차순 정렬
                top_k_checkpoints.sort(key=lambda x: x[0], reverse=True)

                # 리스트 크기가 K개를 초과하면 가장 낮은 점수 모델 제거
                if len(top_k_checkpoints) > top_k:
                    score_to_remove, path_to_remove = top_k_checkpoints.pop() # 마지막 항목 제거
                    print(f"  Removing checkpoint {os.path.basename(path_to_remove)} (score: {score_to_remove:.4f}) as it's no longer in top-{top_k}")
                    if os.path.exists(path_to_remove):
                        try:
                            os.remove(path_to_remove)
                        except Exception as e_rem:
                            print(f"    Error removing file {path_to_remove}: {e_rem}")
                    else:
                        print(f"    Warning: File to remove not found: {path_to_remove}")

            except Exception as e_save:
                print(f"  Error saving checkpoint: {e_save}")

        # 조기 종료는 여전히 '최고 점수' 갱신 여부 기준
        if epoch_val_f1 > max_val_f1:
            print(f"  Validation Macro F1 improved ({max_val_f1:.4f} --> {epoch_val_f1:.4f}).")
            max_val_f1 = epoch_val_f1
            best_epoch = epoch # 최고 점수 에폭 업데이트
            epochs_no_improve = 0 # 카운터 리셋
        else:
            epochs_no_improve += 1 # 최고 점수 갱신 안됨, 카운터 증가
            print(f"  Validation Macro F1 did not improve from the best ({max_val_f1:.4f}). ({epochs_no_improve}/{patience})")

        # 조기 종료 조건 확인
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch+1} after {patience} epochs without improvement from the best F1 score.")
            break # 학습 루프 종료

    print(f"\n학습 완료.")
    print(f"최종 Top-{top_k} 모델 성능 및 경로:")
    if not top_k_checkpoints:
        print("  No models were saved.")
    else:
        for i, (score, path) in enumerate(top_k_checkpoints):
            print(f"  Top {i+1}: Score={score:.4f}, Path={path}")
        # 최고 기록 자체는 여전히 max_val_f1과 best_epoch으로 알 수 있음
        print(f"\nOverall Best Epoch: {best_epoch+1}, Overall Best Validation Macro F1: {max_val_f1:.4f}")

    return history


# --- 3. 설정 변수 정의 ---
MODEL_NAME = 'convnext_large.fb_in22k_ft_in1k_384' # 사용할 모델 이름
NUM_CLASSES = 7        # 분류할 암석 종류 개수
IMG_SIZE = 384         # 모델 입력 이미지 크기

# --- 데이터 경로 설정 ---
# TODO: 실제 데이터 경로로 수정하세요
TRAIN_DATA_DIR = r"/home/metaai2/workspace/limseunghwan/open/train"
VAL_DATA_DIR = r"/home/metaai2/workspace/limseunghwan/open/val"

# --- 학습 하이퍼파라미터 ---
EPOCHS = 20          # 총 학습 에폭 수 (조절 가능)
BATCH_SIZE = 8        # 배치 크기 (GPU 메모리에 맞게 조절)
BASE_LR = 1e-5         # 기본 학습률 (ConvNeXt fine-tuning에 적합한 값으로 시작, 튜닝 필요)
WEIGHT_DECAY = 1e-2    # 가중치 감쇠 (AdamW와 함께 사용)
WARMUP_EPOCHS = 5      # Warmup 에폭 수
GRADIENT_CLIPPING = 1.0 # Gradient Clipping 값 (사용하지 않으려면 None)

# --- 시스템 설정 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = os.cpu_count() // 2  # 사용 가능한 CPU 코어의 절반 정도 (조절 가능)

# --- 저장 경로 설정 ---
SAVE_DIR = './saved_models' # 모델 저장 디렉토리
os.makedirs(SAVE_DIR, exist_ok=True) # 디렉토리 생성
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, f'{MODEL_NAME.split(".")[0]}_best_f1.pth') # 모델 파일 경로

# --- 4. 모델 로드 및 수정 ---
print(f"Loading model: {MODEL_NAME}")
# timm을 사용하여 사전 학습된 ConvNeXt 모델 로드
model = timm.create_model(MODEL_NAME, pretrained=True)

# 모델의 분류기(head) 부분을 새로운 클래스 수에 맞게 교체
# num_ftrs = model.head.in_features # ConvNeXt는 보통 'head' 속성 사용
# model.head = nn.Linear(num_ftrs, NUM_CLASSES)
model.reset_classifier(num_classes=NUM_CLASSES)
print(f"Model head replaced for {NUM_CLASSES} classes.")

# 모델을 지정된 장치로 이동
model = model.to(DEVICE)

# --- 5. 데이터 변환 정의 ---
# 학습 데이터 변환 (Data Augmentation 포함 - 이전 예시 기반)
# TODO: 필요시 scale, rotation, colorjitter 등 세부 파라미터 조절
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.5, 1.0), ratio=(0.75, 1.3333), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15, interpolation=transforms.InterpolationMode.BILINEAR, fill=0),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet 기본값 사용
])

# 검증/테스트 데이터 변환 (Data Augmentation 없음)
# timm 기본 설정을 사용하거나 직접 정의
try:
    # timm 설정을 우선 사용 시도
    config = timm.data.resolve_data_config({}, model=model)
    val_transform = timm.data.create_transform(**config, is_training=False)
    print("Using timm's default validation transform.")
except Exception as e:
    # timm 설정 로드 실패 시 직접 정의
    print(f"Failed to get timm config ({e}), defining validation transform manually.")
    val_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# --- 6. 데이터셋 및 데이터 로더 준비 ---
# TODO: 실제 데이터셋 클래스를 사용하세요. (예: torchvision.datasets.ImageFolder)
# ImageFolder 사용 예시:
from torchvision.datasets import ImageFolder

print(f"Loading datasets from: {TRAIN_DATA_DIR} and {VAL_DATA_DIR}")
try:
    train_dataset = ImageFolder(root=TRAIN_DATA_DIR, transform=train_transform)
    val_dataset = ImageFolder(root=VAL_DATA_DIR, transform=val_transform)

    print(f"Found {len(train_dataset)} training images and {len(val_dataset)} validation images.")
    print(f"Classes: {train_dataset.classes}") # 클래스 이름 확인

    # Focal Loss의 alpha 계산 (선택적) - 클래스 빈도 기반
    class_counts = np.bincount([s[1] for s in train_dataset.samples])
    if len(class_counts) != NUM_CLASSES:
        print(f"Warning: Number of found classes ({len(class_counts)}) does not match NUM_CLASSES ({NUM_CLASSES}). Adjust NUM_CLASSES or check dataset.")
        # focal_loss_alpha = None
    else:
        total_samples = sum(class_counts)
        class_weights = [total_samples / count if count > 0 else 0 for count in class_counts]
        max_weight = max(class_weights) if any(w > 0 for w in class_weights) else 1 # 0으로 나누기 방지
        class_weights = [w / max_weight for w in class_weights]
        focal_loss_alpha = torch.tensor(class_weights, device=DEVICE, dtype=torch.float32)
        print(f"Calculated Focal Loss alpha (normalized): {focal_loss_alpha.cpu().numpy()}")
    # focal_loss_alpha = None # 우선 None으로 설정

except FileNotFoundError:
    print(f"Error: Data directory not found. Please check TRAIN_DATA_DIR and VAL_DATA_DIR.")
    exit()
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# 데이터 로더 생성
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# --- 7. 손실 함수, 옵티마이저, 스케줄러 정의 ---
# 손실 함수 (Focal Loss 또는 CrossEntropyLoss)
criterion = FocalLoss(alpha=focal_loss_alpha, gamma=2.0).to(DEVICE)
# criterion = nn.CrossEntropyLoss().to(DEVICE) # CrossEntropy 사용 시

# 옵티마이저 (AdamW 추천)
optimizer = optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)

# LR 스케줄러 (Cosine Annealing with Warmup)
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=BASE_LR * 0.01)

# --- 8. 학습 및 검증 실행 ---
if __name__ == "__main__": # 스크립트로 실행될 때만 학습 시작
    print("\nStarting training process...")
    model_name_base = MODEL_NAME.split('/')[-1].split('.')[0] if '/' in MODEL_NAME else MODEL_NAME.split('.')[0]
    history = train_and_validate_best_f1(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=EPOCHS,
        device=DEVICE,
        num_classes=NUM_CLASSES,
        save_dir=SAVE_DIR,             # 디렉토리 전달
        model_name_base=model_name_base, # 모델 이름 기반 전달
        top_k=5,                       # 저장할 개수 지정 (예: 5)
        gradient_clipping=GRADIENT_CLIPPING,
        lr_scheduler=lr_scheduler,
        warmup_epochs=WARMUP_EPOCHS,
        base_lr=BASE_LR
    )

    print("\nTraining finished.")
    # 최종 결과 요약 출력 (함수 내에서도 출력됨)
    if history['val_macro_f1_scores']: # 점수가 기록되었는지 확인
        print("Training History Summary:")
        best_f1_overall = max(history['val_macro_f1_scores'])
        best_epoch_overall = history['val_macro_f1_scores'].index(best_f1_overall) + 1
        print(f"  Overall Best Validation Macro F1 in history: {best_f1_overall:.4f}")
        print(f"  Achieved at Epoch: {best_epoch_overall}")
    else:
        print("No validation scores recorded in history.")