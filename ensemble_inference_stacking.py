# ensemble_inference_stacking.py
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
import joblib # 학습된 메타 모델 로딩용

# --- 스태킹을 위한 추가 import ---
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler # MLP 사용 시 특성 스케일링 권장
from timm.data import resolve_data_config, create_transform

# --- 1. 설정 (Configuration) ---
NUM_CLASSES = 7
CLASS_NAMES = ['Andesite', 'Basalt', 'Etc', 'Gneiss', 'Granite', 'Mud_Sandstone', 'Weathered_Rock']
if len(CLASS_NAMES) != NUM_CLASSES:
    raise ValueError(f"CLASS_NAMES의 개수({len(CLASS_NAMES)})가 NUM_CLASSES({NUM_CLASSES})와 일치하지 않습니다.")

# 실제 경로로 수정 필요
TEST_CSV_PATH = r'/home/metaai2/workspace/limseunghwan/open/test.csv'    # test.csv 경로
IMAGE_BASE_DIR = r'/home/metaai2/workspace/limseunghwan/open'           # 이미지 기본 디렉토리
OUTPUT_CSV_DIR = './submissions_ensemble' # 출력 CSV 저장 디렉토리

BATCH_SIZE_INFERENCE = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS_INFERENCE = os.cpu_count() // 2 if os.cpu_count() is not None else 4

MODEL_CONFIGS = [
    {
        "name": "MaxViT_XLarge_384_epoch19",
        "architecture": 'maxvit_xlarge_tf_384.in21k_ft_in1k',
        "saved_path": './saved_models_maxvit_xlarge_384/maxvit_xlarge_tf_384_epoch19_f1_0.9294.pth',
        "img_size": 384,
    },
    {
        "name": "MaxViT_XLarge_384_epoch18",
        "architecture": 'maxvit_xlarge_tf_384.in21k_ft_in1k',
        "saved_path": './saved_models_maxvit_xlarge_384/maxvit_xlarge_tf_384_epoch18_f1_0.9280.pth',
        "img_size": 384,
    },
    {
        "name": "ConvNeXt_Large_384_epoch20",
        "architecture": 'convnext_large.fb_in22k_ft_in1k_384',
        "saved_path": './saved_models_convnext/convnext_large_epoch19_f1_0.9121.pth',
        "img_size": 384,
    },
    {
        "name": "Swin_Large_384_Best",
        "architecture": 'swin_large_patch4_window12_384.ms_in22k_ft_in1k',
        "saved_path": './best_swin_large_model_test_limited_batches.pth',
        "img_size": 384,
    }
]

# --- 스태킹 메타 모델 경로 설정 ---
# 이 파일들은 별도의 학습 과정을 통해 생성되어야 합니다.
META_MODELS_BASE_DIR = './meta_models' # 메타 모델 저장 기본 디렉토리
META_MODEL_LR_PATH = os.path.join(META_MODELS_BASE_DIR, 'logistic_regression_meta_model.joblib')
META_MODEL_LGBM_PATH = os.path.join(META_MODELS_BASE_DIR, 'lightgbm_meta_model.joblib')
META_MODEL_MLP_PATH = os.path.join(META_MODELS_BASE_DIR, 'mlp_meta_model.joblib')
META_SCALER_MLP_PATH = os.path.join(META_MODELS_BASE_DIR, 'mlp_scaler.joblib')


first_model_img_size = MODEL_CONFIGS[0]["img_size"]
for config in MODEL_CONFIGS:
    if config["img_size"] != first_model_img_size:
        print(f"경고: 모델 {config['name']}의 img_size({config['img_size']})가 첫 번째 모델의 img_size({first_model_img_size})와 다릅니다.")
IMG_SIZE = first_model_img_size

# --- 2. 추론용 데이터 변환 ---
print(f"IMG_SIZE={IMG_SIZE}에 대한 추론 변환 정의 중...")
try:
    # 임시 모델 생성 (img_size는 모델에 따라 무시될 수도 있음)
    temp_model = timm.create_model(
        MODEL_CONFIGS[0]["architecture"],
        pretrained=True,
        num_classes=NUM_CLASSES
    )
    
    # 모델에서 config 추출
    data_config = resolve_data_config({}, model=temp_model)

    # 변환 함수 생성
    inference_transform = create_transform(
        **data_config,
        is_training=False
    )

    print(
        f"timm의 기본 변환 사용: "
        f"Mean={data_config.get('mean')}, "
        f"Std={data_config.get('std')}, "
        f"Input_Size={data_config.get('input_size')}"
    )

    del temp_model
except Exception as e:
    print(f"timm 설정 가져오기 실패 ({e}). ImageNet 기본값 사용.")
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
            raise FileNotFoundError(f"{full_img_path}에서 이미지를 찾을 수 없습니다.")
        except Exception as e:
            raise IOError(f"{full_img_path} 이미지를 열 수 없습니다: {e}")
        if self.transform:
            image = self.transform(image)
        return image, relative_img_path

print(f"\nCSV에서 테스트 데이터 로드 중: {TEST_CSV_PATH}")
try:
    test_dataset = TestImageDataset(csv_path=TEST_CSV_PATH, img_dir_root=IMAGE_BASE_DIR, transform=inference_transform)
    if len(test_dataset) == 0:
        print(f"오류: {TEST_CSV_PATH}에서 이미지를 찾을 수 없습니다. 종료합니다.")
        exit()
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE_INFERENCE, shuffle=False, num_workers=NUM_WORKERS_INFERENCE, pin_memory=True)
    print(f"{len(test_dataset)}개의 이미지로 데이터로더를 성공적으로 생성했습니다.")
except FileNotFoundError:
    print(f"오류: {TEST_CSV_PATH}에서 테스트 CSV 파일을 찾을 수 없습니다. 종료합니다.")
    exit()
except Exception as e:
    print(f"오류: 테스트 데이터셋 또는 데이터로더를 생성할 수 없습니다: {e}. 종료합니다.")
    exit()

# --- 4. 모델 로딩 및 예측 함수 ---
def load_model_from_config(model_conf):
    print(f"모델 아키텍처 로드 중: {model_conf['architecture']}")
    architecture_lower = model_conf['architecture'].lower()
    img_size_arg = model_conf.get('img_size', IMG_SIZE) # 모델 설정에 img_size가 없으면 공유 IMG_SIZE 사용

    model_args = {'pretrained': False, 'num_classes': NUM_CLASSES}
    try:
        pretrained_cfg = timm.models.get_pretrained_cfg(model_conf['architecture'])
        input_size = getattr(pretrained_cfg, 'input_size', None)
    except Exception as e:
        input_size = None

    if (
        'maxvit' in architecture_lower 
        or 'efficientformerv2' in architecture_lower 
        or input_size is not None
    ):
        # img_size를 명시적으로 받는 모델이나, 모델 config에 input_size가 정의된 경우
        model_args['img_size'] = img_size_arg

    model = timm.create_model(model_conf['architecture'], **model_args)

    print(f"학습된 가중치 로드 중: {model_conf['saved_path']}")
    if not os.path.exists(model_conf['saved_path']):
        raise FileNotFoundError(f"모델 가중치를 찾을 수 없음: {model_conf['saved_path']}")
    try:
        state_dict = torch.load(model_conf['saved_path'], map_location=torch.device('cpu'))
        if 'state_dict' in state_dict: state_dict = state_dict['state_dict']
        if 'model' in state_dict: state_dict = state_dict['model']
        model.load_state_dict(state_dict)
    except RuntimeError: # 키 불일치 시 'module.' 접두사 시도
        print("키 앞에 'module.' 접두사가 있는지 확인 후 다시 로드 시도 중...")
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
    except Exception as e:
        raise e
    model = model.to(DEVICE)
    model.eval()
    return model

def get_probabilities(model, loader, device, model_name="Model"):
    model.eval()
    all_probs_list = []
    all_original_filenames_list = []
    with torch.no_grad():
        for images_batch, filenames_batch in tqdm(loader, desc=f"{model_name}으로 예측 중"):
            images_batch = images_batch.to(device)
            outputs = model(images_batch)
            probs = torch.softmax(outputs, dim=1)
            all_probs_list.append(probs.cpu().numpy()) # 바로 numpy로 변환
            all_original_filenames_list.extend(list(filenames_batch))
    all_probs_array = np.concatenate(all_probs_list, axis=0)
    return all_probs_array, all_original_filenames_list

# --- 5. 모든 기본 모델에 대한 추론 실행 (메타 특성 생성) ---
if __name__ == "__main__":
    os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)
    os.makedirs(META_MODELS_BASE_DIR, exist_ok=True) # 메타 모델 디렉토리 생성 (없다면)

    print("\n스태킹 추론 프로세스 시작 중...")
    print(f"사용 디바이스: {DEVICE}")

    all_base_model_probs_list = []
    image_paths_from_loader = None
    processed_base_model_names = []

    for i, model_conf in enumerate(MODEL_CONFIGS):
        print(f"\n--- 기본 모델 처리 중: {model_conf['name']} ---")
        try:
            model = load_model_from_config(model_conf)
            probs_np, current_paths = get_probabilities(model, test_loader, DEVICE, model_conf['name'])
            all_base_model_probs_list.append(probs_np)
            processed_base_model_names.append(model_conf['name'])

            if image_paths_from_loader is None:
                image_paths_from_loader = current_paths
            elif image_paths_from_loader != current_paths:
                print("경고: 모델 예측 간 이미지 경로 순서 불일치. 문제가 있을 수 있습니다.") # 심각한 오류일 수 있음
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except FileNotFoundError as e:
            print(f"오류: 모델 {model_conf['name']} ({model_conf['saved_path']}) 가중치 파일을 찾을 수 없습니다: {e}. 이 모델은 제외합니다.")
        except Exception as e:
            print(f"오류: 모델 {model_conf['name']} 처리 중 오류 발생: {e}. 이 모델은 제외합니다.")

    if not all_base_model_probs_list:
        print("오류: 성공적으로 처리된 기본 모델이 없습니다. 종료합니다.")
        exit()

    meta_features = np.concatenate(all_base_model_probs_list, axis=1)
    print(f"\n메타 특성 생성 완료. 형태: {meta_features.shape}")

    # --- 6. 학습된 메타 모델 로드 및 예측 ---
    final_predictions = {} # 각 메타 모델의 예측을 저장할 딕셔너리

    # Logistic Regression
    if os.path.exists(META_MODEL_LR_PATH):
        print(f"\n--- Logistic Regression 메타 모델 예측 ---")
        try:
            meta_model_lr = joblib.load(META_MODEL_LR_PATH)
            lr_preds = meta_model_lr.predict(meta_features)
            final_predictions['LogisticRegression'] = lr_preds
            print("Logistic Regression 예측 완료.")
        except Exception as e:
            print(f"오류: Logistic Regression 메타 모델 로드 또는 예측 실패: {e}")
    else:
        print(f"경고: Logistic Regression 메타 모델 파일({META_MODEL_LR_PATH})을 찾을 수 없습니다.")

    # LightGBM
    if os.path.exists(META_MODEL_LGBM_PATH):
        print(f"\n--- LightGBM 메타 모델 예측 ---")
        try:
            meta_model_lgbm = joblib.load(META_MODEL_LGBM_PATH)
            lgbm_preds = meta_model_lgbm.predict(meta_features)
            final_predictions['LightGBM'] = lgbm_preds
            print("LightGBM 예측 완료.")
        except Exception as e:
            print(f"오류: LightGBM 메타 모델 로드 또는 예측 실패: {e}")
    else:
        print(f"경고: LightGBM 메타 모델 파일({META_MODEL_LGBM_PATH})을 찾을 수 없습니다.")

    # MLP
    if os.path.exists(META_MODEL_MLP_PATH) and os.path.exists(META_SCALER_MLP_PATH):
        print(f"\n--- MLP 메타 모델 예측 ---")
        try:
            meta_model_mlp = joblib.load(META_MODEL_MLP_PATH)
            scaler_mlp = joblib.load(META_SCALER_MLP_PATH)
            scaled_meta_features_mlp = scaler_mlp.transform(meta_features) # 스케일링 적용
            mlp_preds = meta_model_mlp.predict(scaled_meta_features_mlp)
            final_predictions['MLP'] = mlp_preds
            print("MLP 예측 완료.")
        except Exception as e:
            print(f"오류: MLP 메타 모델 로드 또는 예측 실패: {e}")
    else:
        if not os.path.exists(META_MODEL_MLP_PATH):
            print(f"경고: MLP 메타 모델 파일({META_MODEL_MLP_PATH})을 찾을 수 없습니다.")
        if not os.path.exists(META_SCALER_MLP_PATH):
            print(f"경고: MLP 스케일러 파일({META_SCALER_MLP_PATH})을 찾을 수 없습니다.")

    if not final_predictions:
        print("오류: 어떤 메타 모델도 예측을 생성하지 못했습니다. 종료합니다.")
        exit()

    # --- 7. 제출 파일 생성 (각 메타 모델별 또는 추가 앙상블) ---
    print("\n제출 파일 생성 중...")
    original_test_df = pd.read_csv(TEST_CSV_PATH)
    csv_img_path_col = test_dataset.img_path_column

    base_model_names_str = "_".join(sorted([name.split('_')[0] for name in processed_base_model_names]))


    for meta_name, predictions_indices in final_predictions.items():
        predicted_class_names = [CLASS_NAMES[idx] for idx in predictions_indices]

        prediction_map = {
            os.path.normpath(path): pred_label
            for path, pred_label in zip(image_paths_from_loader, predicted_class_names)
        }
        
        # 원본 CSV 순서에 맞게 예측값 매핑
        mapped_predictions = original_test_df[csv_img_path_col].apply(
            lambda x: prediction_map.get(os.path.normpath(x))
        )

        submission_df = pd.DataFrame()
        if 'ID' in original_test_df.columns:
            submission_df['ID'] = original_test_df['ID']
        else: # ID 컬럼이 없다면 파일명에서 ID 추출 (확장자 제외)
            submission_df['ID'] = original_test_df[csv_img_path_col].apply(
                lambda x: os.path.splitext(os.path.basename(x))[0]
            )
        submission_df['rock_type'] = mapped_predictions

        if submission_df['rock_type'].isnull().any():
            num_null = submission_df['rock_type'].isnull().sum()
            print(f"경고 ({meta_name}): CSV의 {num_null}개 이미지가 예측을 받지 못했습니다. 'Etc'로 채웁니다.")
            submission_df['rock_type'].fillna('Etc', inplace=True)

        output_csv_filename = f"submission_stacking_{base_model_names_str}_meta_{meta_name}.csv"
        final_output_csv_path = os.path.join(OUTPUT_CSV_DIR, output_csv_filename)
        submission_df.to_csv(final_output_csv_path, index=False)
        print(f"\n{meta_name} 스태킹 예측 결과 저장 위치: {final_output_csv_path}")
        print(f"{meta_name} 제출 파일 샘플:\n{submission_df.head()}")

    print("\n모든 스태킹 추론 완료.")