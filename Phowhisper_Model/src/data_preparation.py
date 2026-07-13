import os
import pandas as pd
from datasets import Dataset, DatasetDict
from src.config import AUDIO_SCAN_DIR, AUDIO_EXTS, CSV_FILE_COLUMN, CSV_TEXT_COLUMN

def prepare_audio_lookups(scan_dir: str):
    """
    Recursively scans the directory for all audio files and builds lookup indexes
    by basename and stem name.
    
    Args:
        scan_dir (str): Root folder to search for audio files.
        
    Returns:
        tuple: (lookup_by_name, lookup_by_stem) mapping filenames/stems to full absolute paths.
    """
    audio_paths = []
    if os.path.exists(scan_dir):
        for root, _, files in os.walk(scan_dir):
            for f in files:
                if f.lower().endswith(AUDIO_EXTS):
                    audio_paths.append(os.path.abspath(os.path.join(root, f)))
                    
    lookup_by_name = {}
    lookup_by_stem = {}
    for p in audio_paths:
        filename = os.path.basename(p)
        stem = os.path.splitext(filename)[0]
        lookup_by_name[filename] = p
        lookup_by_stem[stem] = p
        
    return lookup_by_name, lookup_by_stem

def make_dataset(df: pd.DataFrame, split_name: str, lookup_by_name: dict, lookup_by_stem: dict) -> Dataset:
    """
    Validates CSV file paths, maps labels, filters out missing audio or empty transcriptions,
    and returns a Hugging Face Dataset.
    """
    df = df.copy()
    
    def resolve_audio_path(file_name):
        file_name = str(file_name).strip()
        filename = os.path.basename(file_name)
        stem = os.path.splitext(filename)[0]
        if filename in lookup_by_name:
            return lookup_by_name[filename]
        if stem in lookup_by_stem:
            return lookup_by_stem[stem]
        return None
        
    # Resolve CSV column entries into absolute local audio paths
    df["audio_path"] = df[CSV_FILE_COLUMN].apply(resolve_audio_path)
    df["text"] = df[CSV_TEXT_COLUMN].astype(str).str.strip()
    
    print(f"[{split_name.upper()} Dataset]")
    print(f"  Total CSV rows: {len(df)}")
    missing_count = df["audio_path"].isna().sum()
    if missing_count > 0:
        print(f"  Warning: Missing audio files for {missing_count} rows.")
        
    # Drop rows that don't have matching local audio files or have empty transcripts
    df = df.dropna(subset=["audio_path"])
    df = df[df["text"] != ""]
    print(f"  Retained samples: {len(df)}")
    
    # Standardize columns to ['audio_path', 'text']
    return Dataset.from_pandas(df[["audio_path", "text"]], preserve_index=False)

def get_dataset_dict(train_csv: str, val_csv: str, test_csv: str) -> DatasetDict:
    """
    Loads train, validation, and test datasets and formats them into a DatasetDict.
    No audio loading or mel-feature extraction is done here (delegated to On-the-Fly collator).
    """
    # Build lookups by walking through the audio directories
    lookup_by_name, lookup_by_stem = prepare_audio_lookups(AUDIO_SCAN_DIR)
    
    datasets = {}
    
    if os.path.exists(train_csv):
        train_df = pd.read_csv(train_csv)
        datasets["train"] = make_dataset(train_df, "train", lookup_by_name, lookup_by_stem)
    if os.path.exists(val_csv):
        val_df = pd.read_csv(val_csv)
        datasets["validation"] = make_dataset(val_df, "validation", lookup_by_name, lookup_by_stem)
    if os.path.exists(test_csv):
        test_df = pd.read_csv(test_csv)
        datasets["test"] = make_dataset(test_df, "test", lookup_by_name, lookup_by_stem)
        
    if not datasets:
        raise ValueError("At least one CSV file must exist to load dataset.")
        
    return DatasetDict(datasets)
