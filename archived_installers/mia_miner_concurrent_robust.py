#!/usr/bin/env python3
"""
MIA GPU Miner - ROBUST Language Detection
Multiple layers of enforcement to ensure correct language response
"""
import os
if os.path.exists("/data"):
    os.environ["HF_HOME"] = "/data/huggingface"
    os.environ["TRANSFORMERS_CACHE"] = "/data/huggingface"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import time
import socket
import logging
import requests
import threading
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
import torch
import psutil
import GPUtil
from flask import Flask, request, jsonify
from waitress import serve
from vllm import LLM, SamplingParams
import re
import unicodedata

# Configure logging
log_file = "/data/miner.log" if os.path.exists("/data") else "miner.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file)
    ]
)
logger = logging.getLogger('mia-concurrent')

# Global model
model = None
app = Flask(__name__)

# Comprehensive language detection patterns
LANGUAGE_PATTERNS = {
    'Spanish': {
        'strong_indicators': ['¿', '¡', 'ñ', 'á', 'é', 'í', 'ó', 'ú'],
        'common_words': ['hola', 'cómo', 'está', 'estás', 'qué', 'cuál', 'cuando', 'donde', 
                        'gracias', 'por favor', 'buenos días', 'buenas tardes', 'buenas noches',
                        'adiós', 'hasta', 'luego', 'mañana', 'necesito', 'quiero', 'puedo'],
        'particles': ['el', 'la', 'los', 'las', 'un', 'una', 'de', 'en', 'con', 'para'],
        'score_multiplier': 3
    },
    'French': {
        'strong_indicators': ['ç', 'à', 'è', 'é', 'ê', 'ë', 'î', 'ï', 'ô', 'ù', 'û', 'ü', 'œ', 'æ'],
        'common_words': ['bonjour', 'bonsoir', 'comment', 'allez', 'vous', 'merci', 'beaucoup',
                        's\'il vous plaît', 'au revoir', 'à bientôt', 'pourquoi', 'quand', 'où'],
        'particles': ['le', 'la', 'les', 'un', 'une', 'de', 'du', 'des', 'à', 'au', 'aux'],
        'score_multiplier': 3
    },
    'German': {
        'strong_indicators': ['ä', 'ö', 'ü', 'Ä', 'Ö', 'Ü', 'ß'],
        'common_words': ['hallo', 'guten tag', 'guten morgen', 'wie', 'geht', 'danke', 'bitte',
                        'auf wiedersehen', 'tschüss', 'warum', 'wann', 'wo', 'ich', 'bin'],
        'particles': ['der', 'die', 'das', 'ein', 'eine', 'von', 'zu', 'mit', 'für', 'auf'],
        'score_multiplier': 3
    },
    'Italian': {
        'strong_indicators': ['à', 'è', 'é', 'ì', 'ò', 'ù'],
        'common_words': ['ciao', 'buongiorno', 'buonasera', 'come', 'stai', 'sta', 'grazie',
                        'prego', 'arrivederci', 'perché', 'quando', 'dove', 'sono'],
        'particles': ['il', 'lo', 'la', 'i', 'gli', 'le', 'un', 'una', 'di', 'da', 'in'],
        'score_multiplier': 3
    },
    'Portuguese': {
        'strong_indicators': ['ã', 'õ', 'ç', 'á', 'à', 'â', 'é', 'ê', 'í', 'ó', 'ô', 'ú'],
        'common_words': ['olá', 'oi', 'bom dia', 'boa tarde', 'boa noite', 'como', 'está',
                        'obrigado', 'obrigada', 'por favor', 'tchau', 'até logo'],
        'particles': ['o', 'a', 'os', 'as', 'um', 'uma', 'de', 'do', 'da', 'em', 'no', 'na'],
        'score_multiplier': 3
    },
    'Chinese': {
        'strong_indicators': ['你', '我', '他', '她', '们', '的', '是', '在', '有', '这', '那',
                             '什么', '怎么', '为什么', '吗', '呢', '吧', '啊'],
        'common_words': ['你好', '谢谢', '对不起', '再见', '请问', '可以', '需要', '想要'],
        'particles': [],
        'score_multiplier': 5
    },
    'Japanese': {
        'strong_indicators': ['あ', 'い', 'う', 'え', 'お', 'か', 'き', 'く', 'け', 'こ',
                             'は', 'ひ', 'ふ', 'へ', 'ほ', 'を', 'ん', 'ア', 'イ', 'ウ'],
        'common_words': ['こんにちは', 'ありがとう', 'すみません', 'さようなら', 'お願い'],
        'particles': ['は', 'が', 'を', 'に', 'で', 'と', 'の', 'か', 'も', 'や'],
        'score_multiplier': 5
    },
    'Korean': {
        'strong_indicators': ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ',
                             'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ', 'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ'],
        'common_words': ['안녕하세요', '감사합니다', '죄송합니다', '안녕히'],
        'particles': ['은', '는', '이', '가', '을', '를', '에', '에서', '와', '과'],
        'score_multiplier': 5
    },
    'Russian': {
        'strong_indicators': list('абвгдеёжзийклмнопрстуфхцчшщъыьэюя'),
        'common_words': ['привет', 'здравствуйте', 'спасибо', 'пожалуйста', 'до свидания'],
        'particles': ['в', 'на', 'с', 'к', 'у', 'от', 'до', 'за', 'под', 'над'],
        'score_multiplier': 4
    },
    'Arabic': {
        'strong_indicators': list('ابتثجحخدذرزسشصضطظعغفقكلمنهوي'),
        'common_words': ['مرحبا', 'شكرا', 'من فضلك', 'آسف', 'وداعا'],
        'particles': ['في', 'على', 'من', 'إلى', 'عن', 'مع'],
        'score_multiplier': 5
    }
}

def detect_language_robust(text):
    """
    Robust language detection using multiple methods
    """
    if not text or not text.strip():
        return 'English', 100  # Default to English with high confidence
    
    text_lower = text.lower()
    scores = {}
    
    # Method 1: Character and word pattern matching
    for language, patterns in LANGUAGE_PATTERNS.items():
        score = 0
        matches = []
        
        # Check strong indicators (special characters)
        for indicator in patterns['strong_indicators']:
            if indicator in text or indicator in text_lower:
                score += 10
                matches.append(f"char:{indicator}")
        
        # Check common words
        for word in patterns['common_words']:
            if re.search(r'\b' + re.escape(word) + r'\b', text_lower):
                score += 15
                matches.append(f"word:{word}")
        
        # Check particles (very common small words)
        for particle in patterns['particles']:
            if re.search(r'\b' + re.escape(particle) + r'\b', text_lower):
                score += 5
                matches.append(f"particle:{particle}")
        
        # Apply language-specific multiplier
        score *= patterns['score_multiplier']
        
        if score > 0:
            scores[language] = (score, matches)
    
    # Method 2: Unicode block detection
    unicode_blocks = {
        'Chinese': ['CJK', 'CHINESE'],
        'Japanese': ['HIRAGANA', 'KATAKANA', 'CJK'],
        'Korean': ['HANGUL', 'CJK'],
        'Arabic': ['ARABIC'],
        'Russian': ['CYRILLIC']
    }
    
    for char in text:
        char_name = unicodedata.name(char, '').upper()
        for language, blocks in unicode_blocks.items():
            if any(block in char_name for block in blocks):
                if language not in scores:
                    scores[language] = (0, [])
                current_score, matches = scores[language]
                scores[language] = (current_score + 20, matches + [f"unicode:{char}"])
    
    # Method 3: Common English patterns (negative scoring for other languages)
    english_patterns = ['the', 'is', 'are', 'was', 'were', 'have', 'has', 'had', 
                       'will', 'would', 'could', 'should', 'what', 'where', 'when',
                       'how', 'why', 'hello', 'hi', 'thanks', 'please']
    
    english_score = 0
    english_matches = []
    for pattern in english_patterns:
        if re.search(r'\b' + pattern + r'\b', text_lower):
            english_score += 10
            english_matches.append(f"word:{pattern}")
    
    # Check for English-only characters (no special chars from other languages)
    if all(ord(char) < 256 or char.isspace() for char in text):
        english_score += 30
        english_matches.append("ascii-only")
    
    if english_score > 0:
        scores['English'] = (english_score, english_matches)
    
    # Determine the language with highest confidence
    if not scores:
        logger.info("No language patterns detected, defaulting to English")
        return 'English', 100
    
    # Get the best match
    best_language = max(scores.items(), key=lambda x: x[1][0])
    language = best_language[0]
    confidence = min(100, best_language[1][0])
    
    # Log detection details
    logger.info(f"Language detection for '{text[:50]}...':")
    for lang, (score, matches) in sorted(scores.items(), key=lambda x: x[1][0], reverse=True):
        logger.info(f"  {lang}: score={score}, matches={matches[:5]}")
    logger.info(f"  => Detected: {language} (confidence: {confidence}%)")
    
    return language, confidence

# CRITICAL: Language-specific system prompts with STRONG enforcement
def get_language_prompt(language, confidence):
    """
    Get a strongly worded system prompt for the detected language
    """
    base_prompts = {
        'English': """You are MIA, a helpful AI assistant. You MUST respond ONLY in English.
CRITICAL RULE: Use ONLY English in your response. Do NOT use any other language.
If you use any non-English words, your response will be rejected.""",
        
        'Spanish': """Eres MIA, un asistente de IA útil. DEBES responder SOLO en español.
REGLA CRÍTICA: Usa SOLO español en tu respuesta. NO uses ningún otro idioma.
Si usas palabras que no sean en español, tu respuesta será rechazada.""",
        
        'French': """Tu es MIA, un assistant IA utile. Tu DOIS répondre UNIQUEMENT en français.
RÈGLE CRITIQUE: Utilise UNIQUEMENT le français dans ta réponse. N'utilise AUCUNE autre langue.
Si tu utilises des mots non français, ta réponse sera rejetée.""",
        
        'German': """Du bist MIA, ein hilfreicher KI-Assistent. Du MUSST NUR auf Deutsch antworten.
KRITISCHE REGEL: Verwende NUR Deutsch in deiner Antwort. Verwende KEINE andere Sprache.
Wenn du nicht-deutsche Wörter verwendest, wird deine Antwort abgelehnt.""",
        
        'Italian': """Sei MIA, un assistente AI utile. DEVI rispondere SOLO in italiano.
REGOLA CRITICA: Usa SOLO l'italiano nella tua risposta. NON usare nessun'altra lingua.
Se usi parole non italiane, la tua risposta sarà rifiutata.""",
        
        'Portuguese': """Você é MIA, um assistente de IA útil. Você DEVE responder APENAS em português.
REGRA CRÍTICA: Use APENAS português em sua resposta. NÃO use nenhum outro idioma.
Se você usar palavras não portuguesas, sua resposta será rejeitada.""",
        
        'Chinese': """你是MIA，一个有用的AI助手。你必须只用中文回答。
关键规则：在你的回答中只使用中文。不要使用任何其他语言。
如果你使用非中文词汇，你的回答将被拒绝。""",
        
        'Japanese': """あなたはMIA、親切なAIアシスタントです。日本語のみで返答しなければなりません。
重要なルール：返答には日本語のみを使用してください。他の言語を使用しないでください。
日本語以外の言葉を使用した場合、あなたの返答は拒否されます。""",
        
        'Korean': """당신은 MIA, 도움이 되는 AI 비서입니다. 반드시 한국어로만 대답해야 합니다.
중요 규칙: 답변에는 한국어만 사용하세요. 다른 언어를 사용하지 마세요.
한국어가 아닌 단어를 사용하면 답변이 거부됩니다.""",
        
        'Russian': """Ты MIA, полезный ИИ-ассистент. Ты ДОЛЖЕН отвечать ТОЛЬКО на русском языке.
КРИТИЧЕСКОЕ ПРАВИЛО: Используй ТОЛЬКО русский язык в своем ответе. НЕ используй никакой другой язык.
Если ты используешь не русские слова, твой ответ будет отклонен.""",
        
        'Arabic': """أنت MIA، مساعد ذكاء اصطناعي مفيد. يجب أن تجيب بالعربية فقط.
قاعدة حرجة: استخدم العربية فقط في إجابتك. لا تستخدم أي لغة أخرى.
إذا استخدمت كلمات غير عربية، سيتم رفض إجابتك."""
    }
    
    prompt = base_prompts.get(language, f"""You are MIA, a helpful AI assistant. 
The user's message was detected as {language} with {confidence}% confidence.
You MUST respond in {language} ONLY. Using any other language is strictly forbidden.""")
    
    # Add extra enforcement for high confidence
    if confidence > 80:
        prompt += f"\n\nFINAL WARNING: The user wrote in {language}. You MUST respond in {language}. No exceptions."
    
    return prompt

# Language-specific response starters to force correct language
LANGUAGE_STARTERS = {
    'English': "",  # No starter needed
    'Spanish': "Por supuesto, ",
    'French': "Bien sûr, ",
    'German': "Natürlich, ",
    'Italian': "Certamente, ",
    'Portuguese': "Claro, ",
    'Chinese': "好的，",
    'Japanese': "はい、",
    'Korean': "네, ",
    'Russian': "Конечно, ",
    'Arabic': "بالطبع، "
}

# Copy all the classes from the original concurrent miner...
class GPUMemoryManager:
    """Manages GPU memory and determines concurrent job capacity"""
    
    def __init__(self):
        self.base_memory_per_job_mb = 500
        self.safety_margin = 0.15
        
    def get_gpu_memory_info(self):
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                return {
                    'total_mb': gpu.memoryTotal,
                    'used_mb': gpu.memoryUsed,
                    'free_mb': gpu.memoryFree,
                    'utilization': gpu.memoryUtil,
                    'temperature': gpu.temperature
                }
        except:
            if torch.cuda.is_available():
                total = torch.cuda.get_device_properties(0).total_memory / 1024**2
                allocated = torch.cuda.memory_allocated(0) / 1024**2
                free = total - allocated
                
                return {
                    'total_mb': total,
                    'used_mb': allocated,
                    'free_mb': free,
                    'utilization': allocated / total,
                    'temperature': 0
                }
        return None
    
    def get_max_concurrent_jobs(self):
        mem_info = self.get_gpu_memory_info()
        if not mem_info:
            return 1
        
        safe_free_mb = mem_info['free_mb'] * (1 - self.safety_margin)
        max_jobs = max(1, int(safe_free_mb / self.base_memory_per_job_mb))
        
        if mem_info['total_mb'] < 8000:
            max_jobs = min(max_jobs, 2)
        elif mem_info['total_mb'] < 16000:
            max_jobs = min(max_jobs, 4)
        else:
            max_jobs = min(max_jobs, 8)
        
        logger.info(f"GPU Memory: {mem_info['free_mb']:.0f}MB free / {mem_info['total_mb']:.0f}MB total")
        logger.info(f"Max concurrent jobs: {max_jobs}")
        
        return max_jobs

class ConcurrentMinerClient:
    def __init__(self, model_server):
        self.backend_url = "https://mia-backend-production.up.railway.app"
        self.local_url = "http://localhost:8000"
        self.miner_name = f"gpu-miner-{socket.gethostname()}-concurrent"
        self.miner_id = None
        self.model_server = model_server
        
        self.job_queue = Queue()
        self.active_jobs = {}
        self.max_workers = 4
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.memory_manager = GPUMemoryManager()
        
        self.jobs_completed = 0
        self.total_tokens = 0
        self.start_time = time.time()
    
    def get_gpu_info(self):
        try:
            if torch.cuda.is_available():
                return {
                    'name': torch.cuda.get_device_name(0),
                    'memory_mb': torch.cuda.get_device_properties(0).total_memory // (1024*1024)
                }
        except:
            pass
        return {'name': 'vLLM-AWQ GPU', 'memory_mb': 8192}
    
    async def fetch_jobs_async(self, session, num_jobs):
        jobs = []
        
        for _ in range(num_jobs):
            try:
                async with session.get(
                    f"{self.backend_url}/get_work",
                    params={"miner_id": self.miner_id},
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        work = await response.json()
                        if work and work.get("request_id"):
                            jobs.append(work)
            except Exception as e:
                logger.error(f"Error fetching job: {e}")
        
        return jobs
    
    def process_job(self, job):
        job_id = job['request_id']
        self.active_jobs[job_id] = job
        
        try:
            logger.info(f"Processing job: {job_id}")
            start_time = time.time()
            
            r = requests.post(
                f"{self.local_url}/generate",
                json={
                    "prompt": job.get("prompt", ""),
                    "max_tokens": job.get("max_tokens", 500),
                    "request_id": job_id
                },
                timeout=120
            )
            
            if r.status_code == 200:
                result = r.json()
                processing_time = time.time() - start_time
                
                submit_data = {
                    "miner_id": self.miner_id,
                    "request_id": job_id,
                    "result": {
                        "response": result.get("text", ""),
                        "tokens_generated": result.get("tokens_generated", 0),
                        "processing_time": processing_time
                    }
                }
                
                submit_r = requests.post(
                    f"{self.backend_url}/submit_result",
                    json=submit_data,
                    timeout=30
                )
                
                if submit_r.status_code == 200:
                    tokens = result.get("tokens_generated", 0)
                    speed = result.get('tokens_per_second', 0)
                    
                    self.jobs_completed += 1
                    self.total_tokens += tokens
                    
                    logger.info(f"✓ Job {job_id} complete ({speed:.1f} tok/s)")
                    
                    if self.jobs_completed % 10 == 0:
                        elapsed = time.time() - self.start_time
                        jobs_per_min = (self.jobs_completed / elapsed) * 60
                        tokens_per_min = (self.total_tokens / elapsed) * 60
                        logger.info(f"Stats: {self.jobs_completed} jobs, {jobs_per_min:.1f} jobs/min, {tokens_per_min:.0f} tokens/min")
                else:
                    logger.error(f"Failed to submit result for {job_id}")
            else:
                logger.error(f"Generation failed for {job_id}")
                
        except Exception as e:
            logger.error(f"Error processing job {job_id}: {e}")
        finally:
            self.active_jobs.pop(job_id, None)
    
    async def job_fetcher(self):
        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    max_concurrent = self.memory_manager.get_max_concurrent_jobs()
                    current_jobs = len(self.active_jobs)
                    capacity = max_concurrent - current_jobs
                    
                    if capacity > 0:
                        jobs = await self.fetch_jobs_async(session, capacity)
                        
                        for job in jobs:
                            self.executor.submit(self.process_job, job)
                        
                        if jobs:
                            logger.info(f"Fetched {len(jobs)} new jobs (active: {len(self.active_jobs)}/{max_concurrent})")
                    
                    await asyncio.sleep(0.5)
                        
                except Exception as e:
                    logger.error(f"Job fetcher error: {e}")
                    await asyncio.sleep(5)
    
    def register(self):
        try:
            gpu_info = self.get_gpu_info()
            
            try:
                ip = requests.get('https://api.ipify.org', timeout=10).text
            except:
                ip = "vastai"
            
            data = {
                "name": self.miner_name,
                "ip_address": ip,
                "gpu_name": f"{gpu_info['name']} (Concurrent)",
                "gpu_memory_mb": gpu_info['memory_mb'],
                "status": "idle"
            }
            
            logger.info(f"Registering miner: {self.miner_name}")
            
            r = requests.post(f"{self.backend_url}/register_miner", json=data, timeout=30)
            
            if r.status_code == 200:
                resp = r.json()
                self.miner_id = resp.get('miner_id')
                logger.info(f"✓ Registered successfully! Miner ID: {self.miner_id}")
                return True
            else:
                logger.error(f"Registration failed: {r.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Registration error: {e}")
            return False
    
    def run(self):
        logger.info("Starting concurrent mining...")
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(self.job_fetcher())
        except KeyboardInterrupt:
            logger.info("Shutdown requested")
        finally:
            self.executor.shutdown(wait=True)
            loop.close()

class ModelServer:
    def __init__(self):
        self.model = None
        self.server_thread = None
        self.request_tracking = {}
    
    def load_model(self):
        global model
        
        logger.info("Loading vLLM with AWQ model for concurrent processing...")
        
        try:
            self.model = LLM(
                model="TheBloke/Mistral-7B-OpenOrca-AWQ",
                quantization="awq",
                dtype="half",
                gpu_memory_utilization=0.85,
                max_model_len=2048,
                max_num_seqs=8
            )
            model = self.model
            
            logger.info("✓ vLLM with AWQ loaded for concurrent processing!")
            
            sampling_params = SamplingParams(temperature=0, max_tokens=50)
            start = time.time()
            outputs = model.generate(["Hello, how are you?"], sampling_params)
            elapsed = time.time() - start
            
            tokens = len(outputs[0].outputs[0].token_ids)
            speed = tokens / elapsed
            logger.info(f"Single request speed: {speed:.1f} tokens/second")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def start_server(self):
        def run_server():
            logger.info("Starting inference server on port 8000...")
            serve(app, host="0.0.0.0", port=8000, threads=8)
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        time.sleep(3)

@app.route("/health", methods=["GET"])
def health():
    mem_info = GPUMemoryManager().get_gpu_memory_info()
    return jsonify({
        "status": "ready" if model is not None else "loading",
        "backend": "vLLM-AWQ-Concurrent-ROBUST",
        "gpu_memory": mem_info
    })

@app.route("/generate", methods=["POST"])
def generate():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503
    
    try:
        data = request.json
        prompt = data.get("prompt", "")
        max_tokens = data.get("max_tokens", 500)
        request_id = data.get("request_id", "unknown")
        
        # ROBUST language detection
        detected_language, confidence = detect_language_robust(prompt)
        
        # Get strongly-worded system prompt
        system_prompt = get_language_prompt(detected_language, confidence)
        
        # Get language starter
        lang_starter = LANGUAGE_STARTERS.get(detected_language, "")
        
        # Format prompt with MAXIMUM enforcement
        formatted_prompt = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
{lang_starter}"""
        
        logger.info(f"Request {request_id} - Language: {detected_language} (confidence: {confidence}%)")
        
        # vLLM sampling parameters - adjusted for better language adherence
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=max_tokens,
            repetition_penalty=1.1,
            stop=["<|im_end|>", "<|im_start|>"],
            # Lower temperature for more consistent language
            presence_penalty=0.1 if confidence > 80 else 0.0
        )
        
        # Generate with vLLM
        start_time = time.time()
        outputs = model.generate([formatted_prompt], sampling_params)
        generation_time = time.time() - start_time
        
        # Extract response
        generated_text = outputs[0].outputs[0].text.strip()
        
        # Remove the language starter if it's still there
        if lang_starter and generated_text.startswith(lang_starter.strip()):
            generated_text = generated_text[len(lang_starter):].strip()
        
        # Clean up any leading colons or spaces
        generated_text = generated_text.lstrip(": ")
        
        # FINAL CHECK: Verify response language (optional but adds robustness)
        # Could add post-processing here to verify the response is in the correct language
        
        tokens_generated = len(outputs[0].outputs[0].token_ids)
        tokens_per_sec = tokens_generated / generation_time if generation_time > 0 else 0
        
        logger.debug(f"Request {request_id}: {tokens_generated} tokens in {generation_time:.2f}s ({tokens_per_sec:.1f} tok/s)")
        
        return jsonify({
            "text": generated_text,
            "tokens_generated": tokens_generated,
            "generation_time": round(generation_time, 2),
            "tokens_per_second": round(tokens_per_sec, 1),
            "request_id": request_id,
            "detected_language": detected_language,
            "language_confidence": confidence
        })
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return jsonify({"error": str(e)}), 500

def main():
    logger.info("=" * 60)
    logger.info("MIA GPU Miner - ROBUST Language Detection")
    logger.info("Multiple enforcement layers for correct language")
    logger.info("=" * 60)
    
    try:
        import GPUtil
    except:
        logger.info("Installing GPUtil for GPU monitoring...")
        os.system("pip install gputil")
    
    model_server = ModelServer()
    
    if not model_server.load_model():
        logger.error("Failed to load model")
        sys.exit(1)
    
    model_server.start_server()
    
    for i in range(30):
        try:
            r = requests.get("http://localhost:8000/health", timeout=5)
            if r.status_code == 200:
                logger.info("✓ Model server ready")
                break
        except:
            pass
        time.sleep(2)
    
    miner = ConcurrentMinerClient(model_server)
    
    attempts = 0
    while not miner.register():
        attempts += 1
        if attempts > 5:
            logger.error("Failed to register after 5 attempts")
            sys.exit(1)
        logger.info(f"Retrying registration in 30s... (attempt {attempts}/5)")
        time.sleep(30)
    
    try:
        miner.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()