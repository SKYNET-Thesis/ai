import re

def normalize_text(text: str) -> str:
    """
    Standardizes Vietnamese transcription text for speech evaluation (WER/CER).
    Converts to lowercase, removes punctuation/special characters, and strips excess spaces.
    """
    text = str(text).lower().strip()
    # Replace non-word and non-Vietnamese characters with spaces
    text = re.sub(
        r"[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]",
        " ",
        text
    )
    # Collapse multiple whitespaces into a single space
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_audio(audio_path: str, target_sr: int = 16000, mono: bool = True) -> tuple:
    """
    Loads an audio file and resamples it to target_sr.
    Attempts librosa.load first, falling back to PyAV (av) if librosa fails.
    """
    # Try librosa first (good for standard formats, if backend is available)
    try:
        import librosa
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            audio_array, sr = librosa.load(audio_path, sr=target_sr, mono=mono)
            return audio_array, sr
    except Exception as e:
        # Fallback to PyAV
        pass

    # If librosa failed or backend was missing, use PyAV
    try:
        import av
        import numpy as np

        container = av.open(audio_path)
        if not container.streams.audio:
            raise ValueError(f"No audio streams found in {audio_path}")
        stream = container.streams.audio[0]

        resampler = av.AudioResampler(
            format='fltp',
            layout='mono' if mono else stream.layout.name,
            rate=target_sr
        )

        audio_frames = []
        for frame in container.decode(stream):
            resampled_frames = resampler.resample(frame)
            for rf in resampled_frames:
                arr = rf.to_ndarray()
                if mono:
                    arr = arr.flatten()
                audio_frames.append(arr)

        if not audio_frames:
            raise ValueError(f"Failed to decode any audio frames from {audio_path}")

        audio_array = np.concatenate(audio_frames)
        return audio_array, target_sr
    except Exception as av_err:
        raise RuntimeError(
            f"Failed to load audio file {audio_path} using both librosa and PyAV.\n"
            f"Librosa error: {e if 'e' in locals() else 'unknown'}\n"
            f"PyAV error: {av_err}"
        )

