import os
import sys
import argparse
import torch
import librosa
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline
from src.utils import load_audio

def transcribe(audio_path, processor, model, device):
    # Load audio at 16kHz mono
    print(f"Loading audio file: {audio_path}...")
    audio_array, sr = load_audio(audio_path, target_sr=16000, mono=True)
    
    # Create ASR pipeline with chunking
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        device=device,
        ignore_warning=True
    )
    
    print("Transcribing (running inference)...")
    result = pipe(audio_array)
    
    return result["text"].strip()

def main():
    # Force sys.stdout to output UTF-8 encoding
    sys.stdout.reconfigure(encoding='utf-8')
    
    parser = argparse.ArgumentParser(description="Test inference on a single audio file using fine-tuned PhoWhisper.")
    parser.add_argument("--model_dir", type=str, default="models/phowhisper_finetuned", help="Path to the fine-tuned model folder.")
    parser.add_argument("--audio_path", type=str, default="", help="Path to your audio file (mp3, wav, flac, m4a).")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device.upper()}")
    
    # Load model and processor
    if not os.path.exists(args.model_dir):
        print(f"Error: Model directory not found at {args.model_dir}. Did you run training successfully?")
        return
        
    print(f"Loading model from {args.model_dir}...")
    processor = AutoProcessor.from_pretrained(args.model_dir)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model_dir,
        low_cpu_mem_usage=True
    ).to(device)
    model.eval()
    
    audio_path = args.audio_path
    if not audio_path:
        # Prompt user if not provided in arguments
        audio_path = input("Please enter the path to your audio file: ").strip()
        
    # Strip quotes if copied from file path explorer
    if audio_path.startswith(('"', "'")) and audio_path.endswith(('"', "'")):
        audio_path = audio_path[1:-1]
        
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at '{audio_path}'")
        return
        
    try:
        text = transcribe(audio_path, processor, model, device)
        print("\n" + "="*50)
        print("TRANSCRIPTION:")
        print(text)
        print("="*50 + "\n")
    except Exception as e:
        import traceback
        print("An error occurred during inference:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
