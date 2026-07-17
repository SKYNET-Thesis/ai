import os

# Base Model Selection
MODEL_NAME = "vinai/PhoWhisper-small"

# Dataset Directories and Files
DATA_DIR = "data"
AUDIO_SCAN_DIR = DATA_DIR
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
VAL_CSV = os.path.join(DATA_DIR, "val.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")

# Allowed audio extensions
AUDIO_EXTS = (".mp3", ".wav", ".flac", ".m4a", ".mp4")

# Vietnamese CSV Columns mapping
CSV_FILE_COLUMN = "tên file"
CSV_TEXT_COLUMN = "transcript"

# Output directories
OUTPUT_DIR = "models/phowhisper_finetuned"
LOGS_DIR = "outputs/logs"
EVAL_RESULTS_FILE = "outputs/evaluation_results.json"

# Training Hyperparameters (Aligned with Kaggle Notebook)
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16
LEARNING_RATE = 5e-6
WARMUP_STEPS = 50
EPOCHS = 1
FP16 = True

# Inference and Decoding configurations
GENERATION_MAX_NEW_TOKENS = 225
