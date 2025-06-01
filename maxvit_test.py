# MaxViT_inference.py
import torch
import torch.nn as nn # Not strictly needed for inference if model is loaded, but good practice
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm
import pandas as pd
from PIL import Image
import os
from tqdm import tqdm
import numpy as np # For potential use, e.g. if needing to map class indices to names manually

# --- 1. Configuration (Adjust these paths and parameters) ---
# Model and Data Parameters (should match how the loaded model was trained)
MODEL_ARCHITECTURE_NAME = 'maxvit_xlarge_tf_384.in21k_ft_in1k' # Architecture of your saved model
NUM_CLASSES = 7        # Number of classes your model was trained on
IMG_SIZE = 384         # Image size your model was trained with

# --- Paths ---
# TODO: IMPORTANT! Update this to the path of YOUR TRAINED MaxViT model .pth file
SAVED_MODEL_PATH = './saved_models_maxvit_xlarge_384/maxvit_xlarge_tf_384_epoch19_f1_0.9294.pth' # EXAMPLE PATH!
TEST_CSV_PATH = r'/home/metaai2/workspace/limseunghwan/open/test.csv'    # Path to your test.csv
IMAGE_BASE_DIR = r'/home/metaai2/workspace/limseunghwan/open'           # Base directory for images in test.csv
OUTPUT_CSV_DIR = './submissions_maxvit' # Directory to save the output CSV
# OUTPUT_CSV_PATH will be generated dynamically based on model name.

# --- Inference Parameters ---
BATCH_SIZE_INFERENCE = 4 # Adjust based on your GPU memory (MaxViT XLarge is demanding)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS_INFERENCE = os.cpu_count() // 2 if os.cpu_count() else 4

# --- Class Names (CRITICAL!) ---
# TODO: ***매우 중요*** 당신의 데이터셋에 맞는 실제 클래스 이름을 순서대로 정의하세요.
#       이 순서는 모델이 학습될 때 클래스에 할당된 인덱스와 일치해야 합니다.
#       (예: train_dataset.classes 에서 가져온 순서)
CLASS_NAMES = ['Andesite', 'Basalt', 'Etc', 'Gneiss', 'Granite', 'Mud_Sandstone', 'Weathered_Rock'] # 예시 값, 반드시 수정!

if len(CLASS_NAMES) != NUM_CLASSES:
    raise ValueError(f"Number of CLASS_NAMES ({len(CLASS_NAMES)}) does not match NUM_CLASSES ({NUM_CLASSES}). Please check your configuration.")

# --- 2. Model Loading ---
print(f"Loading model architecture: {MODEL_ARCHITECTURE_NAME}")
# Create the model structure (pretrained=False because we load our own weights)
model = timm.create_model(
    MODEL_ARCHITECTURE_NAME,
    pretrained=False,
    num_classes=NUM_CLASSES,
    img_size=IMG_SIZE  # Ensure img_size is passed if model supports/requires it at creation
)

print(f"Loading trained weights from: {SAVED_MODEL_PATH}")
if not os.path.exists(SAVED_MODEL_PATH):
    print(f"ERROR: Model weights file not found at {SAVED_MODEL_PATH}. Please check the path.")
    exit()

try:
    # Load the state dictionary
    model.load_state_dict(torch.load(SAVED_MODEL_PATH, map_location=DEVICE))
    print(f"Successfully loaded model weights onto {DEVICE}.")
except Exception as e:
    print(f"ERROR: Could not load model weights: {e}")
    exit()

model = model.to(DEVICE)
model.eval() # Set the model to evaluation mode
print("Model ready for inference.")

# --- 3. Data Transformations for Inference ---
# Re-create the validation/inference transform used during training.
# It's best if this matches exactly what was used for the validation set.
print("Defining inference transform...")
try:
    # Attempt to use timm's recommended settings for the loaded model
    # Pass the loaded model instance to resolve_data_config
    config = timm.data.resolve_data_config({}, model=model)
    # Override input_size to be sure it matches your training
    config['input_size'] = (3, IMG_SIZE, IMG_SIZE) # (C, H, W)
    
    inference_transform = timm.data.create_transform(**config, is_training=False)
    print(f"Using timm's default transform based on loaded model config: Mean={config['mean']}, Std={config['std']}")
except Exception as e:
    print(f"Failed to get timm config for inference model ({e}). Defining transform manually using ImageNet defaults.")
    inference_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet defaults
    ])
print(f"Inference Transform: {inference_transform}")


# --- 4. Custom Dataset for Test Images ---
class TestImageDataset(Dataset):
    def __init__(self, csv_path, img_dir_root, transform=None):
        self.data_frame = pd.read_csv(csv_path)
        self.img_dir_root = img_dir_root
        self.transform = transform
        # Determine the image path column name
        if 'img_path' not in self.data_frame.columns:
            print(f"Warning: 'img_path' column not found in {csv_path}. Assuming first column ('{self.data_frame.columns[0]}') contains image paths.")
            self.img_path_column = self.data_frame.columns[0]
        else:
            self.img_path_column = 'img_path'
        print(f"Using column '{self.img_path_column}' from CSV for image paths.")

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        relative_img_path = self.data_frame.loc[idx, self.img_path_column]
        full_img_path = os.path.join(self.img_dir_root, relative_img_path)

        try:
            image = Image.open(full_img_path).convert('RGB')
        except FileNotFoundError:
            print(f"ERROR: Image not found at {full_img_path}. Check IMAGE_BASE_DIR and CSV paths.")
            # To prevent crashing the DataLoader, one might return a placeholder or skip.
            # For now, raising an error is fine to highlight the issue.
            raise
        except Exception as e:
            print(f"ERROR: Could not open image {full_img_path}: {e}")
            raise

        if self.transform:
            image = self.transform(image)

        return image, relative_img_path # Return image and its original path for mapping

# --- 5. DataLoader for Test Set ---
print(f"Loading test data from CSV: {TEST_CSV_PATH}")
print(f"Image base directory: {IMAGE_BASE_DIR}")
try:
    test_dataset = TestImageDataset(csv_path=TEST_CSV_PATH,
                                    img_dir_root=IMAGE_BASE_DIR,
                                    transform=inference_transform)
    if len(test_dataset) == 0:
        print(f"ERROR: No images found or loaded from {TEST_CSV_PATH}. Please check the CSV and image paths.")
        exit()
        
    test_loader = DataLoader(test_dataset,
                             batch_size=BATCH_SIZE_INFERENCE,
                             shuffle=False, # No need to shuffle for inference
                             num_workers=NUM_WORKERS_INFERENCE,
                             pin_memory=True)
    print(f"Successfully created DataLoader with {len(test_dataset)} images for testing.")
except FileNotFoundError:
    print(f"ERROR: Test CSV file not found at {TEST_CSV_PATH}. Please check the path.")
    exit()
except Exception as e:
    print(f"ERROR: Could not create test dataset or DataLoader: {e}")
    exit()

# --- 6. Prediction Function ---
def predict_on_test_data(model_to_predict, loader, device, class_names_map):
    model_to_predict.eval() # Ensure model is in eval mode
    all_predicted_indices = []
    all_original_filenames = [] # To store the original filenames/paths from CSV

    with torch.no_grad(): # Disable gradient calculations for inference
        for images_batch, filenames_batch in tqdm(loader, desc="Predicting"):
            images_batch = images_batch.to(device)
            
            outputs = model_to_predict(images_batch)
            _, predicted_indices_batch = torch.max(outputs, 1) # Get the index of the max log-probability

            all_predicted_indices.extend(predicted_indices_batch.cpu().numpy())
            all_original_filenames.extend(list(filenames_batch)) # filenames_batch is a tuple of strings

    # Convert predicted indices to actual class names
    predicted_class_names_list = [class_names_map[idx] for idx in all_predicted_indices]
    
    return all_original_filenames, predicted_class_names_list

# --- 7. Run Inference and Save Results ---
if __name__ == "__main__":
    print("\nStarting inference process...")

    # Generate output CSV path
    os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)
    model_filename_base = os.path.splitext(os.path.basename(SAVED_MODEL_PATH))[0]
    output_csv_filename = f"submission_{model_filename_base}.csv"
    final_output_csv_path = os.path.join(OUTPUT_CSV_DIR, output_csv_filename)

    original_paths_from_loader, string_predictions = predict_on_test_data(
        model,
        test_loader,
        DEVICE,
        CLASS_NAMES
    )

    # Create a DataFrame for submission
    # Ensure the order of predictions matches the order of images in the original test.csv
    
    # Load the original test CSV to get the correct order and 'ID' column format if needed
    original_test_df = pd.read_csv(TEST_CSV_PATH)
    
    # Create a mapping from the image paths returned by DataLoader to their predictions
    # Use os.path.normpath to handle potential path separator differences (e.g. / vs \)
    prediction_map = {
        os.path.normpath(path): pred_label 
        for path, pred_label in zip(original_paths_from_loader, string_predictions)
    }

    # Map predictions back to the original CSV's image paths
    # This ensures that even if DataLoader reorders (it shouldn't with shuffle=False),
    # or if there are missing images handled gracefully, the mapping is correct.
    csv_img_path_col = test_dataset.img_path_column # Get the column name used for image paths
    
    mapped_predictions = original_test_df[csv_img_path_col].apply(
        lambda x: prediction_map.get(os.path.normpath(x))
    )

    # Create the submission DataFrame
    submission_df = pd.DataFrame()
    # The 'ID' column in submission usually requires the filename without extension
    submission_df['ID'] = original_test_df[csv_img_path_col].apply(
        lambda x: os.path.splitext(os.path.basename(x))[0]
    )
    submission_df['rock_type'] = mapped_predictions # Use the mapped predictions

    # Check for any images that didn't get a prediction (should not happen if all images load)
    if submission_df['rock_type'].isnull().any():
        num_null = submission_df['rock_type'].isnull().sum()
        print(f"WARNING: {num_null} images from the CSV did not receive a prediction. "
              "This might be due to missing image files or errors during loading. "
              "Consider filling NaNs if appropriate for the submission.")
        # Example: Fill with a default class if needed (e.g., the most frequent one or 'Etc')
        # default_class_for_nan = CLASS_NAMES[0] # Or any other logic
        # submission_df['rock_type'].fillna(default_class_for_nan, inplace=True)
        # print(f"Filled {num_null} NaNs with '{default_class_for_nan}'.")


    # Save the submission file
    submission_df.to_csv(final_output_csv_path, index=False)
    print(f"\nInference complete. Predictions saved to: {final_output_csv_path}")
    print(f"Sample of the submission file:\n{submission_df.head()}")