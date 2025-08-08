#!/usr/bin/env python3
"""
MIA GPU Miner - Concurrent with Simple Language Detection
Uses keyword-based detection without external dependencies
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

# Language detection patterns
LANGUAGE_PATTERNS = {
    'Spanish': {
        'words': ['hola', 'cómo', 'está', 'estás', 'qué', 'por favor', 'gracias', 'bueno', 'buena', 
                  'sí', 'no', 'cuando', 'donde', 'quien', 'porque', 'necesito', 'quiero', 'puedo',
                  'tiempo', 'día', 'año', 'hacer', 'tener', 'decir', 'más', 'muy', 'también'],
        'chars': ['ñ', '¿', '¡', 'á', 'é', 'í', 'ó', 'ú']
    },
    'French': {
        'words': ['bonjour', 'bonsoir', 'comment', 'ça va', 'merci', 'beaucoup', 'très', 'bien',
                  's\'il vous plaît', 'je', 'tu', 'il', 'elle', 'nous', 'vous', 'ils', 'elles',
                  'être', 'avoir', 'faire', 'aller', 'pour', 'dans', 'avec', 'sans', 'mais'],
        'chars': ['ç', 'à', 'è', 'é', 'ê', 'ë', 'î', 'ï', 'ô', 'ù', 'û', 'ü', 'ÿ', 'æ', 'œ']
    },
    'German': {
        'words': ['hallo', 'guten tag', 'wie', 'geht', 'danke', 'bitte', 'sehr', 'gut',
                  'ich', 'du', 'er', 'sie', 'es', 'wir', 'ihr', 'Sie', 'der', 'die', 'das',
                  'ein', 'eine', 'haben', 'sein', 'werden', 'können', 'müssen', 'wollen'],
        'chars': ['ä', 'ö', 'ü', 'Ä', 'Ö', 'Ü', 'ß']
    },
    'Italian': {
        'words': ['ciao', 'buongiorno', 'buonasera', 'come', 'stai', 'sta', 'grazie', 'prego',
                  'molto', 'bene', 'io', 'tu', 'lui', 'lei', 'noi', 'voi', 'loro', 'essere',
                  'avere', 'fare', 'andare', 'dire', 'vedere', 'sapere', 'dare', 'volere'],
        'chars': ['à', 'è', 'é', 'ì', 'ò', 'ù']
    },
    'Portuguese': {
        'words': ['olá', 'oi', 'bom dia', 'boa tarde', 'boa noite', 'como', 'está', 'obrigado',
                  'obrigada', 'por favor', 'sim', 'não', 'eu', 'tu', 'você', 'ele', 'ela',
                  'nós', 'vós', 'eles', 'elas', 'ser', 'estar', 'ter', 'fazer', 'ir'],
        'chars': ['ã', 'õ', 'ç', 'á', 'à', 'â', 'é', 'ê', 'í', 'ó', 'ô', 'ú']
    },
    'Chinese': {
        'chars': ['你', '我', '他', '她', '们', '的', '是', '在', '有', '这', '那', '个', '了',
                  '不', '会', '好', '吗', '什么', '怎么', '为什么', '谢谢', '对不起', '请']
    },
    'Japanese': {
        'chars': ['あ', 'い', 'う', 'え', 'お', 'か', 'き', 'く', 'け', 'こ', 'さ', 'し', 'す',
                  'せ', 'そ', 'た', 'ち', 'つ', 'て', 'と', 'な', 'に', 'ぬ', 'ね', 'の', 'は',
                  'ひ', 'ふ', 'へ', 'ほ', 'ま', 'み', 'む', 'め', 'も', 'や', 'ゆ', 'よ', 'ら',
                  'り', 'る', 'れ', 'ろ', 'わ', 'を', 'ん', 'ア', 'イ', 'ウ', 'エ', 'オ']
    },
    'Korean': {
        'chars': ['안녕', '하세요', '감사', '합니다', '죄송', '합니다', '네', '아니요', '예']
    },
    'Russian': {
        'chars': ['а', 'б', 'в', 'г', 'д', 'е', 'ё', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н',
                  'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь',
                  'э', 'ю', 'я', 'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ё', 'Ж', 'З', 'И', 'Й']
    },
    'Arabic': {
        'chars': ['ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص',
                  'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ي']
    }
}

def detect_language_simple(text):
    """Simple language detection based on keywords and character patterns"""
    text_lower = text.lower()
    
    # Count matches for each language
    scores = {}
    
    for language, patterns in LANGUAGE_PATTERNS.items():
        score = 0
        
        # Check for character matches
        if 'chars' in patterns:
            for char in patterns['chars']:
                if char in text or char.lower() in text_lower:
                    score += 2  # Character matches are strong indicators
        
        # Check for word matches
        if 'words' in patterns:
            for word in patterns['words']:
                if word in text_lower:
                    score += 3  # Word matches are very strong indicators
                elif re.search(r'\b' + re.escape(word) + r'\b', text_lower):
                    score += 3
        
        if score > 0:
            scores[language] = score
    
    # Return the language with highest score, default to English
    if scores:
        detected = max(scores.items(), key=lambda x: x[1])[0]
        logger.info(f"Language detection scores: {scores} -> {detected}")
        return detected
    else:
        logger.info("No language patterns detected, defaulting to English")
        return 'English'

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
        "backend": "vLLM-AWQ-Concurrent",
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
        
        # Detect language using simple keyword detection
        detected_language = detect_language_simple(prompt)
        
        # Create language-specific system prompt with strong reinforcement
        language_instructions = {
            'English': "You are MIA, a helpful AI assistant. You must respond ONLY in English. This is critical - use English only.",
            'Spanish': "Eres MIA, un asistente de IA útil. DEBES responder SOLO en español. Esto es crítico - usa solo español.",
            'French': "Tu es MIA, un assistant IA utile. Tu DOIS répondre UNIQUEMENT en français. C'est critique - utilise uniquement le français.",
            'Chinese': "你是MIA，一个有用的AI助手。你必须只用中文回答。这很重要 - 只使用中文。",
            'German': "Du bist MIA, ein hilfreicher KI-Assistent. Du MUSST NUR auf Deutsch antworten. Das ist kritisch - verwende nur Deutsch.",
            'Italian': "Sei MIA, un assistente AI utile. DEVI rispondere SOLO in italiano. Questo è critico - usa solo l'italiano.",
            'Portuguese': "Você é MIA, um assistente de IA útil. Você DEVE responder APENAS em português. Isso é crítico - use apenas português.",
            'Russian': "Ты MIA, полезный ИИ-ассистент. Ты ДОЛЖЕН отвечать ТОЛЬКО на русском языке. Это критично - используй только русский.",
            'Japanese': "あなたはMIA、親切なAIアシスタントです。日本語のみで返答しなければなりません。これは重要です - 日本語のみを使用してください。",
            'Korean': "당신은 MIA, 도움이 되는 AI 비서입니다. 반드시 한국어로만 대답해야 합니다. 이것은 중요합니다 - 한국어만 사용하세요.",
            'Arabic': "أنت MIA، مساعد ذكاء اصطناعي مفيد. يجب أن تجيب بالعربية فقط. هذا مهم - استخدم العربية فقط."
        }
        
        # Get the appropriate system prompt
        system_prompt = language_instructions.get(detected_language, 
            f"You are MIA, a helpful AI assistant. The user's message appears to be in {detected_language}. You MUST respond ONLY in {detected_language}. Do not use any other language under any circumstances.")
        
        # Add extra reinforcement
        system_prompt += f"\n\nCRITICAL INSTRUCTION: Respond ONLY in {detected_language}. Using any other language is completely forbidden."
        
        # Format prompt with strong language enforcement
        formatted_prompt = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant"""
        
        logger.info(f"Request {request_id} - Detected language: {detected_language}")
        
        # vLLM sampling parameters
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=max_tokens,
            repetition_penalty=1.1,
            stop=["<|im_end|>", "<|im_start|>"]
        )
        
        # Generate with vLLM
        start_time = time.time()
        outputs = model.generate([formatted_prompt], sampling_params)
        generation_time = time.time() - start_time
        
        # Extract response
        generated_text = outputs[0].outputs[0].text.strip()
        generated_text = generated_text.lstrip(": ")
        
        tokens_generated = len(outputs[0].outputs[0].token_ids)
        tokens_per_sec = tokens_generated / generation_time if generation_time > 0 else 0
        
        logger.debug(f"Request {request_id}: {tokens_generated} tokens in {generation_time:.2f}s ({tokens_per_sec:.1f} tok/s)")
        
        return jsonify({
            "text": generated_text,
            "tokens_generated": tokens_generated,
            "generation_time": round(generation_time, 2),
            "tokens_per_second": round(tokens_per_sec, 1),
            "request_id": request_id,
            "detected_language": detected_language
        })
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return jsonify({"error": str(e)}), 500

def main():
    logger.info("=" * 60)
    logger.info("MIA GPU Miner - Concurrent (Simple Language Detection)")
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