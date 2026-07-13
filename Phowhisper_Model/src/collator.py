import torch
import librosa
from dataclasses import dataclass
from typing import Any, Dict, List

@dataclass
class OnTheFlySpeechCollator:
    """
    On-the-fly speech collator that loads audio files dynamically during training batch collation.
    Saves RAM and pre-training cache space.
    """
    processor: Any
    sampling_rate: int = 16000

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audio_arrays = []
        for f in features:
            audio_array, sr = librosa.load(
                f["audio_path"],
                sr=self.sampling_rate,
                mono=True
            )
            audio_arrays.append(audio_array)
            
        # 1. Process waveforms into spectrograms
        batch = self.processor.feature_extractor(
            audio_arrays,
            sampling_rate=self.sampling_rate,
            return_tensors="pt"
        )
        
        # 2. Tokenize transcription texts
        labels_batch = self.processor.tokenizer(
            [f["text"] for f in features],
            padding=True,
            return_tensors="pt"
        )
        
        # 3. Replace pad token IDs with -100 to ignore in loss calculation
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1),
            -100
        )
        
        batch["labels"] = labels
        
        return batch
