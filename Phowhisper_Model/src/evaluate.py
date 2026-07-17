import os
import gc
import json
import argparse
import torch
import librosa
import evaluate
from tqdm import tqdm

# Local imports
from src.config import TEST_CSV, OUTPUT_DIR, EVAL_RESULTS_FILE
from src.model import load_asr_model
from src.data_preparation import get_dataset_dict
from src.utils import normalize_text, load_audio

# Load metrics
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

def transcribe(audio_path: str, processor, model, device) -> str:
    """
    Transcribes a single audio file using loaded model and processor.
    """
    try:
        # Load audio at 16kHz mono
        audio_array, sr = load_audio(audio_path, target_sr=16000, mono=True)
    except Exception as e:
        print(f"Error loading audio file {audio_path}: {e}")
        return ""

    # Extract log-mel spectrogram features
    inputs = processor.feature_extractor(
        audio_array,
        sampling_rate=16000,
        return_tensors="pt"
    )
    input_features = inputs.input_features.to(device)

    # Autoregressive decoding/generation
    with torch.no_grad():
        predicted_ids = model.generate(input_features, max_new_tokens=225)
        
    # Decode to text
    text = processor.tokenizer.batch_decode(
        predicted_ids,
        skip_special_tokens=True
    )[0]
    
    return text.strip()

def clear_model(processor, model):
    """
    Deletes model variables and triggers PyTorch/OS garbage collection
    to prevent GPU Out Of Memory (OOM) errors.
    """
    del processor
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def evaluate_model(model_dir: str, model_name: str, n_samples: int = None, test_csv: str = TEST_CSV):
    """
    Loads test dataset, runs transcribing on n_samples, computes WER and CER,
    and returns metrics dictionary.
    """
    # 1. Load ASR Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device.upper()} for evaluation.")
    processor, model = load_asr_model(model_dir, device)

    # 2. Get Test dataset
    print(f"Loading test dataset from: {test_csv}")
    dataset_dict = get_dataset_dict(
        train_csv="",
        val_csv="",
        test_csv=test_csv
    )
    test_dataset = dataset_dict["test"]

    # Limit sample count if n_samples is specified
    total_samples = len(test_dataset)
    if n_samples is not None and n_samples < total_samples:
        print(f"Limiting evaluation to first {n_samples} samples (out of {total_samples}).")
        eval_range = range(n_samples)
    else:
        print(f"Evaluating all {total_samples} samples.")
        eval_range = range(total_samples)

    predictions = []
    references = []

    # 3. Predict loop
    print("Running batch predictions...")
    for i in tqdm(eval_range):
        sample = test_dataset[i]
        pred = transcribe(sample["audio_path"], processor, model, device)
        ref = sample["text"]

        # Run text normalization (lowercase, stripping punctuation)
        predictions.append(normalize_text(pred))
        references.append(normalize_text(ref))

    # Calculate WER and CER metrics
    wer = wer_metric.compute(predictions=predictions, references=references)
    cer = cer_metric.compute(predictions=predictions, references=references)

    print("\n" + "=" * 50)
    print(f"EVALUATION SUMMARY: {model_name}")
    print(f"Model Directory: {model_dir}")
    print(f"Word Error Rate (WER): {wer:.4f} ({wer * 100:.2f}%)")
    print(f"Character Error Rate (CER): {cer:.4f} ({cer * 100:.2f}%)")
    print("=" * 50 + "\n")

    # Clean memory immediately
    clear_model(processor, model)

    return {
        "model_name": model_name,
        "model_dir": model_dir,
        "samples_evaluated": len(eval_range),
        "wer": wer,
        "cer": cer
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned PhoWhisper model.")
    parser.add_argument("--model_dir", type=str, default=OUTPUT_DIR, help="Path to your model folder.")
    parser.add_argument("--model_name", type=str, default="phowhisper_finetuned", help="Display name for the model.")
    parser.add_argument("--test_csv", type=str, default=TEST_CSV, help="CSV file for test dataset.")
    parser.add_argument("--n_samples", type=int, default=None, help="Limit number of evaluation samples.")
    parser.add_argument("--output_json", type=str, default=EVAL_RESULTS_FILE, help="Path to save evaluation metrics.")
    
    args = parser.parse_args()

    results = evaluate_model(
        model_dir=args.model_dir,
        model_name=args.model_name,
        n_samples=args.n_samples,
        test_csv=args.test_csv
    )

    # Save results to json file
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"Evaluation results successfully saved to: {args.output_json}")

if __name__ == "__main__":
    main()
