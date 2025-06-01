# MaxViT.py 
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
    max_val_f1 = 0.0
    best_epoch = -1
    top_k_checkpoints = []

    f1_metric = torchmetrics.F1Score(task='multiclass', num_classes=num_classes, average='macro').to(device)
    patience = 5 # 조기 종료를 위한 인내 에폭 수
    epochs_no_improve = 0

    os.makedirs(save_dir, exist_ok=True)

    print(f"학습 시작: 총 {epochs} 에폭, Device: {device}")
    print(f"Top-{top_k} 모델 저장 디렉토리: {save_dir}")
    print(f"평가 기준: Validation Macro F1 Score")

    for epoch in range(epochs):
        epoch_start_time = time.time()
        current_lr = optimizer.param_groups[0]['lr']

        if epoch < warmup_epochs:
            warmup_factor = (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = base_lr * warmup_factor
            current_lr = optimizer.param_groups[0]['lr']
        elif lr_scheduler is not None:
             if epoch == warmup_epochs: # 첫 스케줄러 스텝 전 LR을 base_lr로 설정
                 for param_group in optimizer.param_groups:
                     param_group['lr'] = base_lr
                 current_lr = optimizer.param_groups[0]['lr'] # 업데이트된 LR 반영
             lr_scheduler.step() # warmup이 끝난 후 매 에폭마다 스케줄러 호출
             current_lr = optimizer.param_groups[0]['lr']


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

        is_top_k = len(top_k_checkpoints) < top_k or epoch_val_f1 > top_k_checkpoints[-1][0]

        if is_top_k:
            checkpoint_filename = f"{model_name_base}_epoch{epoch+1}_f1_{epoch_val_f1:.4f}.pth"
            checkpoint_path = os.path.join(save_dir, checkpoint_filename)
            try:
                torch.save(model.state_dict(), checkpoint_path)
                print(f"  Checkpoint saved to {checkpoint_path}")
                top_k_checkpoints.append((epoch_val_f1, checkpoint_path))
                top_k_checkpoints.sort(key=lambda x: x[0], reverse=True)
                if len(top_k_checkpoints) > top_k:
                    score_to_remove, path_to_remove = top_k_checkpoints.pop()
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

        if epoch_val_f1 > max_val_f1:
            print(f"  Validation Macro F1 improved ({max_val_f1:.4f} --> {epoch_val_f1:.4f}).")
            max_val_f1 = epoch_val_f1
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"  Validation Macro F1 did not improve from the best ({max_val_f1:.4f}). ({epochs_no_improve}/{patience})")

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch+1} after {patience} epochs without improvement from the best F1 score.")
            break

    print(f"\n학습 완료.")
    print(f"최종 Top-{top_k} 모델 성능 및 경로:")
    if not top_k_checkpoints:
        print("  No models were saved.")
    else:
        for i, (score, path) in enumerate(top_k_checkpoints):
            print(f"  Top {i+1}: Score={score:.4f}, Path={path}")
    print(f"\nOverall Best Epoch: {best_epoch+1 if best_epoch != -1 else 'N/A'}, Overall Best Validation Macro F1: {max_val_f1:.4f}")

    return history

# --- 3. 설정 변수 정의 ---
MODEL_NAME = 'maxvit_xlarge_tf_384.in21k_ft_in1k' # <<< MaxViT XLarge 모델 이름
NUM_CLASSES = 7
IMG_SIZE = 384 # MaxViT 모델 이름에 384가 명시되어 있으므로 이 해상도 사용

# --- 데이터 경로 설정 ---
# TODO: 실제 데이터 경로로 수정하세요
TRAIN_DATA_DIR = r"/home/metaai2/workspace/limseunghwan/open/train" # 예시 경로
VAL_DATA_DIR = r"/home/metaai2/workspace/limseunghwan/open/val"     # 예시 경로

# --- 학습 하이퍼파라미터 ---
EPOCHS = 20 # 에폭 수 (필요에 따라 조절)
# !!! 경고: MaxViT XLarge는 매우 큰 모델입니다. 배치 크기를 작게 시작하세요. !!!
# GPU 메모리가 부족하면 OOM 오류가 발생합니다. (예: 24GB VRAM 에서도 2~4 정도가 한계일 수 있음)
BATCH_SIZE = 2 # 배치 크기 (GPU 메모리에 맞게 조절, MaxViT-XLarge는 매우 많은 메모리를 사용)
BASE_LR = 1e-5 # 기본 학습률 (Fine-tuning에 적합한 값으로 시작)
WEIGHT_DECAY = 1e-2 # 가중치 감쇠 (AdamW와 함께 사용)
WARMUP_EPOCHS = 5   # Warmup 에폭 수
GRADIENT_CLIPPING = 1.0 # Gradient Clipping 값 (사용하지 않으려면 None)

# --- 시스템 설정 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = os.cpu_count() // 2 if os.cpu_count() else 4 # 사용 가능한 CPU 코어의 절반 정도

# --- 저장 경로 설정 ---
SAVE_DIR = f'./saved_models_maxvit_xlarge_384' # 모델 저장 디렉토리 (모델별로 구분)
os.makedirs(SAVE_DIR, exist_ok=True)

# --- 4. 모델 로드 및 수정 ---
print(f"Loading model: {MODEL_NAME}")
model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=NUM_CLASSES, img_size=IMG_SIZE)

# 모델의 분류기(head) 부분을 새로운 클래스 수에 맞게 교체
# timm 모델들은 대부분 reset_classifier를 지원합니다.
print(f"Model classifier replaced for {NUM_CLASSES} classes.")

# 모델을 지정된 장치로 이동
model = model.to(DEVICE)

# --- 5. 데이터 변환 정의 ---
# MaxViT에 대한 일반적인 학습 변환
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.5, 1.0), ratio=(0.75, 1.3333), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15, interpolation=transforms.InterpolationMode.BILINEAR, fill=0),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    # Normalize는 timm.data.resolve_data_config에서 가져온 값으로 설정하는 것이 좋습니다.
    # 여기서는 val_transform과 일관성을 위해 timm 기본값을 사용합니다.
    # 만약 val_transform이 다른 값을 사용한다면 여기도 맞춰야 합니다.
    # 이 예제에서는 val_transform이 timm config를 사용하므로, train도 맞춰줍니다 (아래서 mean/std를 가져옴).
])

try:
    config = timm.data.resolve_data_config({}, model=model) # 새 모델에 맞는 config 로드
    val_transform = timm.data.create_transform(**config, is_training=False)
    print(f"Using timm's default validation transform for {MODEL_NAME}.")
    print(f"Timm default input size for {MODEL_NAME}: {config['input_size']}")
    print(f"Timm default mean: {config['mean']}, std: {config['std']}")

    # 사용자가 지정한 IMG_SIZE와 timm config의 input_size가 다를 수 있으므로, 명시적으로 맞춰줍니다.
    # config['input_size']는 (C, H, W) 형태이므로, H, W는 config['input_size'][-2:] 입니다.
    # 여기서는 IMG_SIZE를 사용합니다.
    print(f"Overriding to use specified IMG_SIZE: {IMG_SIZE} for validation transform consistency.")
    val_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.BICUBIC), # 또는 (config['input_size'][-2], config['input_size'][-1])
        transforms.CenterCrop(IMG_SIZE), # 또는 config['input_size'][-1]
        transforms.ToTensor(),
        transforms.Normalize(mean=config['mean'], std=config['std']) # 모델별 mean/std 사용
    ])
    
    # train_transform의 Normalize도 timm config 값으로 업데이트
    train_transform.transforms.append(transforms.Normalize(mean=config['mean'], std=config['std']))
    # 기존 Normalize가 있었다면 제거하고 새로 추가해야 하지만, 위에서는 ToTensor 뒤에 바로 추가되므로 괜찮음
    # 만약 기존 Normalize(ImageNet 기본값)가 train_transform에 있었다면, 다음과 같이 교체:
    # train_transform.transforms[-1] = transforms.Normalize(mean=config['mean'], std=config['std'])


except Exception as e:
    print(f"Failed to get timm config ({e}), defining validation transform manually (using ImageNet defaults).")
    val_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # train_transform에도 ImageNet 기본값 Normalize 추가
    train_transform.transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))


# --- 6. 데이터셋 및 데이터 로더 준비 ---
from torchvision.datasets import ImageFolder

print(f"Loading datasets from: {TRAIN_DATA_DIR} and {VAL_DATA_DIR}")
try:
    train_dataset = ImageFolder(root=TRAIN_DATA_DIR, transform=train_transform)
    val_dataset = ImageFolder(root=VAL_DATA_DIR, transform=val_transform)

    print(f"Found {len(train_dataset)} training images and {len(val_dataset)} validation images.")
    print(f"Classes: {train_dataset.classes}")

    class_counts = np.bincount([s[1] for s in train_dataset.samples])
    focal_loss_alpha = None # 기본값
    if len(class_counts) == NUM_CLASSES:
        total_samples = sum(class_counts)
        class_weights_raw = [total_samples / count if count > 0 else 0 for count in class_counts]
        
        max_weight = max(class_weights_raw) if any(w > 0 for w in class_weights_raw) else 1
        if max_weight > 0:
            class_weights_normalized = [w / max_weight for w in class_weights_raw]
            focal_loss_alpha = torch.tensor(class_weights_normalized, device=DEVICE, dtype=torch.float32)
            print(f"Calculated Focal Loss alpha (normalized): {focal_loss_alpha.cpu().numpy()}")
        else:
            print("Warning: All class counts are zero. Cannot calculate Focal Loss alpha.")
    else:
        print(f"Warning: Number of found classes ({len(class_counts)}) in dataset does not match NUM_CLASSES ({NUM_CLASSES}). Focal Loss alpha set to None.")

except FileNotFoundError:
    print(f"Error: Data directory not found. Please check TRAIN_DATA_DIR and VAL_DATA_DIR.")
    exit()
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# --- 7. 손실 함수, 옵티마이저, 스케줄러 정의 ---
if focal_loss_alpha is not None:
    criterion = FocalLoss(alpha=focal_loss_alpha, gamma=2.0).to(DEVICE)
    print("Using Focal Loss with calculated alpha.")
else:
    criterion = FocalLoss(gamma=2.0).to(DEVICE)
    print("Using Focal Loss without alpha (or CrossEntropyLoss if preferred).")


optimizer = optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=BASE_LR * 0.01 if EPOCHS > WARMUP_EPOCHS else BASE_LR)


# --- 8. 학습 및 검증 실행 ---
if __name__ == "__main__":
    print(f"\nStarting training process with {MODEL_NAME}...")
    print(f"Batch Size: {BATCH_SIZE}. If you encounter OOM errors, try reducing it further.")
    
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
        save_dir=SAVE_DIR,
        model_name_base=model_name_base,
        top_k=5,
        gradient_clipping=GRADIENT_CLIPPING,
        lr_scheduler=lr_scheduler,
        warmup_epochs=WARMUP_EPOCHS,
        base_lr=BASE_LR
    )

    print("\nTraining finished.")
    if history['val_macro_f1_scores']:
        print("Training History Summary:")
        best_f1_overall = max(history['val_macro_f1_scores'])
        best_epoch_overall = history['val_macro_f1_scores'].index(best_f1_overall) + 1
        print(f"  Overall Best Validation Macro F1 in history: {best_f1_overall:.4f}")
        print(f"  Achieved at Epoch: {best_epoch_overall}")
    else:
        print("No validation scores recorded in history.")