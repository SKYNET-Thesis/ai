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
