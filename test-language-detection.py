#!/usr/bin/env python3
"""Test language detection locally"""

import sys
import os

# Install langdetect if needed
try:
    from langdetect import detect, LangDetectException
except ImportError:
    print("Installing langdetect...")
    os.system("pip install langdetect")
    from langdetect import detect, LangDetectException

# Language mappings
LANGUAGE_MAP = {
    'en': 'English',
    'es': 'Spanish', 
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'ja': 'Japanese',
    'ko': 'Korean',
    'zh-cn': 'Chinese',
    'zh-tw': 'Chinese',
    'ar': 'Arabic',
    'hi': 'Hindi',
    'nl': 'Dutch',
    'pl': 'Polish',
    'tr': 'Turkish',
    'vi': 'Vietnamese',
    'th': 'Thai',
    'id': 'Indonesian',
    'ms': 'Malay'
}

def test_detection(text):
    """Test language detection on a text"""
    try:
        lang_code = detect(text)
        language = LANGUAGE_MAP.get(lang_code, f'Unknown ({lang_code})')
        print(f'Text: "{text}"')
        print(f'Detected: {lang_code} -> {language}')
        print('-' * 50)
        return language
    except Exception as e:
        print(f'Error: {e}')
        return 'English'

# Test cases
test_cases = [
    "Hello how are you",
    "Hola, ¿cómo estás?",
    "Bonjour, comment allez-vous?",
    "Guten Tag, wie geht es Ihnen?",
    "Ciao, come stai?",
    "Olá, como você está?",
    "Привет, как дела?",
    "こんにちは、元気ですか？",
    "안녕하세요, 어떻게 지내세요?",
    "你好，你好吗？",
    "مرحبا، كيف حالك؟",
    "What is the weather like today?",
    "¿Cuál es el clima hoy?",
    "Quel temps fait-il aujourd'hui?",
]

print("Testing language detection...")
print("=" * 50)

for text in test_cases:
    test_detection(text)

# Interactive mode
if len(sys.argv) > 1:
    user_text = ' '.join(sys.argv[1:])
    print("\nUser input:")
    test_detection(user_text)