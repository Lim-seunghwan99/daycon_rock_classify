# ensemble_inference.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm
import pandas as pd
from PIL import Image
import os
from tqdm import tqdm
import numpy as np

# --- 1. 설정 (Configuration) ---
# --- 공통 파라미터 ---
NUM_CLASSES = 7
CLASS_NAMES = ['Andesite', 'Basalt', 'Etc', 'Gneiss', 'Granite', 'Mud_Sandstone', 'Weathered_Rock']
if len(CLASS_NAMES) != NUM_CLASSES:
    raise ValueError(f"CLASS_NAMES의 개수({len(CLASS_NAMES)})가 NUM_CLASSES({NUM_CLASSES})와 일치하지 않습니다.")

TEST_CSV_PATH = r'/home/metaai2/workspace/limseunghwan/open/test.csv'
IMAGE_BASE_DIR = r'/home/metaai2/workspace/limseunghwan/open'
OUTPUT_CSV_DIR = './submissions_ensemble'

# --- 추론 파라미터 ---
BATCH_SIZE_INFERENCE = 4
# <<<--- 변경점 시작 --->>>
# 특정 CUDA 장치 (예: cuda:1) 사용 설정
TARGET_CUDA_DEVICE_ID = 1 # 사용하고자 하는 GPU ID (0부터 시작)

if torch.cuda.is_available():
    if TARGET_CUDA_DEVICE_ID < torch.cuda.device_count():
        DEVICE = torch.device(f"cuda:{TARGET_CUDA_DEVICE_ID}")
        print(f"타겟 CUDA 장치 ID {TARGET_CUDA_DEVICE_ID} ({torch.cuda.get_device_name(TARGET_CUDA_DEVICE_ID)})를 사용합니다.")
    else:
        print(f"경고: 타겟 CUDA 장치 ID {TARGET_CUDA_DEVICE_ID}를 사용할 수 없습니다. 사용 가능한 GPU는 {torch.cuda.device_count()}개 입니다.")
        print(f"사용 가능한 첫 번째 CUDA 장치(cuda:0)를 대신 사용합니다.")
        DEVICE = torch.device("cuda:0")
else:
    print("CUDA를 사용할 수 없습니다. CPU를 사용합니다.")
    DEVICE = torch.device("cpu")
# <<<--- 변경점 끝 --->>>

NUM_WORKERS_INFERENCE = os.cpu_count() // 2 if os.cpu_count() is not None else 4

# --- 모델별 설정 ---
MODEL_CONFIGS = [
    {
        "name": "MaxViT_XLarge_384_epoch19",
        "architecture": 'maxvit_xlarge_tf_384.in21k_ft_in1k',
        "saved_path": './saved_models_maxvit_xlarge_384/maxvit_xlarge_tf_384_epoch19_f1_0.9294.pth',
        "img_size": 384,
        "weight": 0.78
    },
    {
        "name": "ConvNeXt_Large_384_epoch20",
        "architecture": 'convnext_large.fb_in22k_ft_in1k_384',
        "saved_path": './saved_models_convnext/convnext_large_epoch19_f1_0.9121.pth',
        "img_size": 384,
        "weight": 0.53
    },
    {
        "name": "Swin_Large_384_Best",
        "architecture": 'swin_large_patch4_window12_384.ms_in22k_ft_in1k',
        "saved_path": './best_swin_large_model_test_limited_batches.pth',
        "img_size": 384,
        "weight": 0.53
    }
]

# --- 무결성 검사 ---
first_model_img_size = MODEL_CONFIGS[0]["img_size"]
for config_item in MODEL_CONFIGS: # 변수명 config가 함수와 충돌 방지
    if config_item["img_size"] != first_model_img_size:
        print(f"경고: 모델 {config_item['name']}의 img_size({config_item['img_size']})가 첫 번째 모델의 img_size({first_model_img_size})와 다릅니다. "
              f"데이터로더는 {first_model_img_size}를 사용합니다. 이로 인해 {config_item['name']}의 성능이 저하될 수 있습니다.")
IMG_SIZE = first_model_img_size

# --- 2. 추론용 데이터 변환 ---
print(f"IMG_SIZE={IMG_SIZE}에 대한 추론 변환 정의 중 (기준: {MODEL_CONFIGS[0]['name']})...")
try:
    temp_model_for_transform = timm.create_model( # 변수명 변경
        MODEL_CONFIGS[0]["architecture"],
        pretrained=True,
        num_classes=NUM_CLASSES,
        # img_size는 MaxViT의 경우 timm.create_model에서 받을 수 있으나,
        # architecture 문자열에 크기가 명시된 경우(예: _384) timm이 자동으로 인식하기도 함.
        # 명시적으로 전달하려면 아래처럼 하거나, MODEL_CONFIGS[0]에 img_size를 사용.
        img_size=MODEL_CONFIGS[0]['img_size'] if 'maxvit' in MODEL_CONFIGS[0]['architecture'].lower() else None
    )
    data_config_resolved = timm.data.resolve_data_config({}, model=temp_model_for_transform) # 변수명 변경
    inference_transform = timm.data.create_transform(**data_config_resolved, is_training=False)
    print(f"timm의 기본 변환 사용: Mean={data_config_resolved['mean']}, Std={data_config_resolved['std']}, Input_Size={data_config_resolved['input_size']}")
    del temp_model_for_transform
except Exception as e:
    print(f"첫 번째 모델에 대한 timm 설정 가져오기 실패 ({e}). ImageNet 기본값을 사용하여 수동으로 변환 정의 중.")
    inference_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
print(f"공유 추론 변환: {inference_transform}")


# --- 3. 테스트 이미지를 위한 사용자 정의 데이터셋 및 데이터로더 ---
class TestImageDataset(Dataset):
    def __init__(self, csv_path, img_dir_root, transform=None):
        self.data_frame = pd.read_csv(csv_path)
        self.img_dir_root = img_dir_root
        self.transform = transform
        if 'img_path' not in self.data_frame.columns:
            print(f"경고: {csv_path}에서 'img_path' 열을 찾을 수 없습니다. 첫 번째 열('{self.data_frame.columns[0]}')에 이미지 경로가 있다고 가정합니다.")
            self.img_path_column = self.data_frame.columns[0]
        else:
            self.img_path_column = 'img_path'
        print(f"CSV에서 이미지 경로에 '{self.img_path_column}' 열 사용 중.")

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        relative_img_path = self.data_frame.loc[idx, self.img_path_column]
        full_img_path = os.path.join(self.img_dir_root, relative_img_path)
        try:
            image = Image.open(full_img_path).convert('RGB')
        except FileNotFoundError:
            print(f"오류: {full_img_path}에서 이미지를 찾을 수 없습니다.")
            raise
        except Exception as e:
            print(f"오류: {full_img_path} 이미지를 열 수 없습니다: {e}")
            raise
        if self.transform:
            image = self.transform(image)
        return image, relative_img_path

print(f"\nCSV에서 테스트 데이터 로드 중: {TEST_CSV_PATH}")
try:
    test_dataset = TestImageDataset(csv_path=TEST_CSV_PATH,
                                    img_dir_root=IMAGE_BASE_DIR,
                                    transform=inference_transform)
    if len(test_dataset) == 0:
        print(f"오류: {TEST_CSV_PATH}에서 이미지를 찾을 수 없습니다.")
        exit()
    test_loader = DataLoader(test_dataset,
                             batch_size=BATCH_SIZE_INFERENCE,
                             shuffle=False,
                             num_workers=NUM_WORKERS_INFERENCE,
                             pin_memory=True)
    print(f"{len(test_dataset)}개의 이미지로 데이터로더를 성공적으로 생성했습니다.")
except FileNotFoundError:
    print(f"오류: {TEST_CSV_PATH}에서 테스트 CSV 파일을 찾을 수 없습니다.")
    exit()
except Exception as e:
    print(f"오류: 테스트 데이터셋 또는 데이터로더를 생성할 수 없습니다: {e}")
    exit()


# --- 4. 모델 로딩 및 예측 함수 ---
def load_model_from_config(model_conf):
    print(f"모델 아키텍처 로드 중: {model_conf['architecture']}")
    architecture_lower = model_conf['architecture'].lower()

    model_args = {'pretrained': False, 'num_classes': NUM_CLASSES}
    if 'maxvit' in architecture_lower :
        model_args['img_size'] = model_conf['img_size']
    # ConvNeXt, Swin 등은 아키텍처 이름에 크기가 포함되어 img_size 명시 불필요한 경우 많음
    
    model = timm.create_model(model_conf['architecture'], **model_args)

    print(f"학습된 가중치 로드 중: {model_conf['saved_path']}")
    if not os.path.exists(model_conf['saved_path']):
        print(f"오류: 모델 가중치 파일 {model_conf['saved_path']}을(를) 찾을 수 없습니다.")
        raise FileNotFoundError(f"모델 가중치를 찾을 수 없음: {model_conf['saved_path']}")
    
    # CPU로 먼저 로드 후 GPU로 이동 (메모리 문제 완화 및 유연성)
    state_dict = torch.load(model_conf['saved_path'], map_location=torch.device('cpu'))
    
    # 중첩된 state_dict 처리 (예: 'model' 또는 'state_dict' 키)
    if isinstance(state_dict, dict): # Optimizer 등 다른 정보와 함께 저장된 경우
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'model' in state_dict:
            state_dict = state_dict['model']
        # 추가적인 키 패턴이 있다면 여기에 elif로 추가

    try:
        model.load_state_dict(state_dict)
        print(f"{model_conf['name']}의 가중치를 성공적으로 로드했습니다 (원본 키).")
    except RuntimeError as e: # 키 불일치 시 'module.' 접두사 시도
        print(f"원본 키로 로드 실패: {e}. 'module.' 접두사 제거 후 재시도 중...")
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        try:
            model.load_state_dict(new_state_dict)
            print(f"{model_conf['name']}의 가중치('module.' 제거 후)를 성공적으로 로드했습니다.")
        except Exception as e2:
            print(f"오류: 'module.' 접두사 제거 후에도 {model_conf['name']}의 가중치를 로드할 수 없습니다: {e2}")
            raise e # 원본 오류를 다시 발생시켜 문제 파악 용이하게 함
            
    model = model.to(DEVICE) # DEVICE 변수에 설정된 GPU로 모델 이동
    model.eval()
    return model

def get_probabilities(model, loader, device, model_name="Model"):
    model.eval()
    all_probs_list = []
    all_original_filenames_list = []

    with torch.no_grad():
        for images_batch, filenames_batch in tqdm(loader, desc=f"{model_name}으로 예측 중"):
            images_batch = images_batch.to(device) # DEVICE 변수에 설정된 GPU로 배치 이동
            outputs = model(images_batch)
            probs = torch.softmax(outputs, dim=1)
            all_probs_list.append(probs.cpu()) # 계산 후에는 CPU로 옮겨 메모리 확보
            all_original_filenames_list.extend(list(filenames_batch))

    all_probs_tensor = torch.cat(all_probs_list, dim=0)
    return all_probs_tensor.numpy(), all_original_filenames_list


# --- 5. 모든 모델에 대한 추론 실행 및 앙상블 ---
if __name__ == "__main__":
    os.makedirs(OUTPUT_CSV_DIR, exist_ok=True) # 출력 디렉토리 생성
    print("\n앙상블 추론 프로세스 시작 중...")
    # DEVICE 변수는 이미 위에서 설정됨 (예: cuda:1 또는 cuda:0 또는 cpu)
    print(f"사용 디바이스: {DEVICE} ({torch.cuda.get_device_name(DEVICE) if DEVICE.type == 'cuda' else 'CPU'})")


    all_model_probs = []
    model_weights = []
    image_paths_from_loader = None
    processed_model_configs = []

    for i, model_conf_item in enumerate(MODEL_CONFIGS): # 변수명 model_conf가 함수와 충돌 방지
        print(f"\n--- 모델 처리 중: {model_conf_item['name']} ---")
        try:
            model = load_model_from_config(model_conf_item)
            probs_np, current_paths = get_probabilities(model, test_loader, DEVICE, model_conf_item['name'])
            all_model_probs.append(probs_np)
            processed_model_configs.append(model_conf_item)

            if 'weight' in model_conf_item:
                model_weights.append(model_conf_item['weight'])
            else:
                model_weights.append(1.0)

            if image_paths_from_loader is None:
                image_paths_from_loader = current_paths
            elif image_paths_from_loader != current_paths:
                print("경고: 모델 예측 간 이미지 경로 순서 불일치. 문제가 있을 수 있습니다.")
            
            del model
            if DEVICE.type == 'cuda': # CUDA 사용 시에만 캐시 비우기
                torch.cuda.empty_cache()
        except FileNotFoundError as e:
            print(f"오류: 모델 {model_conf_item['name']}의 가중치 파일을 찾을 수 없습니다: {e}. 이 모델은 앙상블에서 제외합니다.")
        except Exception as e:
            print(f"오류: 모델 {model_conf_item['name']} 처리 중 오류 발생: {e}. 이 모델은 앙상블에서 제외합니다.")


    if not all_model_probs:
        print("오류: 성공적으로 처리된 모델이 없습니다. 앙상블을 생성할 수 없습니다. 종료합니다.")
        exit()

    # --- 앙상블 예측 (소프트 보팅, 선택적 가중치 사용) ---
    print("\n소프트 보팅을 사용하여 예측 앙상블 중...")

    if sum(model_weights) == 0 and len(model_weights) > 0 :
        print("경고: 총 가중치가 0이지만 처리된 모델이 있습니다. 동일한 가중치를 사용합니다.")
        normalized_weights = [1.0/len(all_model_probs)] * len(all_model_probs)
    elif sum(model_weights) > 0 :
        total_weight = sum(model_weights)
        normalized_weights = [w / total_weight for w in model_weights]
    else:
        print("경고: 가중치 설정에 문제가 있습니다. 동일한 가중치를 사용합니다.")
        normalized_weights = [1.0/len(all_model_probs)] * len(all_model_probs)


    ensembled_probs_sum = np.zeros_like(all_model_probs[0])
    print("앙상블에 사용된 모델 및 가중치:")
    for i, probs_array in enumerate(all_model_probs):
        print(f"- 모델: {processed_model_configs[i]['name']}, 정규화된 가중치: {normalized_weights[i]:.4f}")
        ensembled_probs_sum += probs_array * normalized_weights[i]

    final_predicted_indices = np.argmax(ensembled_probs_sum, axis=1)
    ensembled_class_names = [CLASS_NAMES[idx] for idx in final_predicted_indices]

    # --- 6. 제출 파일 생성 및 저장 ---
    print("\n제출 파일 준비 중...")
    original_test_df = pd.read_csv(TEST_CSV_PATH)

    prediction_map = {
        os.path.normpath(path): pred_label
        for path, pred_label in zip(image_paths_from_loader, ensembled_class_names)
    }

    csv_img_path_col_name = test_dataset.img_path_column # 변수명 명확히
    mapped_predictions = original_test_df[csv_img_path_col_name].apply(
        lambda x: prediction_map.get(os.path.normpath(x))
    )

    submission_df = pd.DataFrame()
    if 'ID' in original_test_df.columns:
        submission_df['ID'] = original_test_df['ID']
    else:
        submission_df['ID'] = original_test_df[csv_img_path_col_name].apply(
            lambda x: os.path.splitext(os.path.basename(x))[0]
        )
    submission_df['rock_type'] = mapped_predictions

    if submission_df['rock_type'].isnull().any():
        num_null = submission_df['rock_type'].isnull().sum()
        print(f"경고: CSV의 {num_null}개 이미지가 예측을 받지 못했습니다. ")
        # submission_df['rock_type'].fillna('Etc', inplace=True)
        # print(f"{num_null}개의 NaN을 'Etc'로 채웠습니다.")

    os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)
    
    ensemble_model_names_list = []
    for cfg_item in processed_model_configs: # 변수명 변경
        # 이름 정제 로직 (필요에 따라 수정)
        name_parts = cfg_item["name"].split('_')
        cleaned_name = "_".join(name_parts[:min(len(name_parts), 3)]) # 앞 3개 파트 또는 전체
        ensemble_model_names_list.append(cleaned_name)

    ensemble_model_names_str = "_".join(sorted(list(set(ensemble_model_names_list))))

    if not ensemble_model_names_str:
        ensemble_model_names_str = "ENSEMBLE_FAILED"
    elif len(processed_model_configs) < len(MODEL_CONFIGS): # MODEL_CONFIGS는 전체 설정
         ensemble_model_names_str += "_PARTIAL"

    output_csv_filename = f"submission_ensemble_{ensemble_model_names_str}_0528.csv"
    final_output_csv_path = os.path.join(OUTPUT_CSV_DIR, output_csv_filename)

    submission_df.to_csv(final_output_csv_path, index=False)
    print(f"\n앙상블 추론 완료. 예측 결과 저장 위치: {final_output_csv_path}")
    print(f"제출 파일 샘플:\n{submission_df.head()}")