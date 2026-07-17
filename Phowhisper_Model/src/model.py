import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

def load_asr_model(model_dir: str, device: str = None):
    """
    Loads Whisper/PhoWhisper model and processor.
    Sets up mandatory configuration flags for training and evaluation.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    print(f"Loading processor/tokenizer from: {model_dir}")
    processor = AutoProcessor.from_pretrained(model_dir)
    
    print(f"Loading model weight checkpoints from: {model_dir}")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_dir,
        low_cpu_mem_usage=True
    )
    
    # Configure gradient checkpointing and caching properties
    model.config.use_cache = False
    
    # Disable token forcing config to predict language/task natively
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []
    
    # Matching settings on model config
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = None
    
    # Move to targeted computing device
    model = model.to(device)
    model.eval()
    
    return processor, model
