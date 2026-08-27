# src/llamafactory/multilingual_prompt/prompt_texts.py

from typing import Dict


LANG_FULL_NAME: Dict[str, str] = {
    "en": "English",
    "zh": "Chinese",
    "de": "German",
    "ru": "Russian",
    "cs": "Czech",
    "my": "Burmese",
    "bur": "Burmese",
    "th": "Thai",
    "vi": "Vietnamese",
    "km": "Khmer",
    "lo": "Lao",
}


def get_full_name(lang_code: str) -> str:
    """
    返回语言代码对应的全英文名称。
    若找不到则原样返回语言代码。
    """
    return LANG_FULL_NAME.get(lang_code.lower().strip(), lang_code)
