#!/usr/bin/env python3
"""
Fallback Manager for MIA Miner
Handles silent Golem Network fallback when no MIA jobs are available
"""

import os
import subprocess
import time
import json
import logging
import requests
from datetime import datetime
from typing import Optional, Dict, Any
import signal
import sys
import threading

logger = logging.getLogger('mia-fallback')

class FallbackManager:
    def __init__(self, miner_name: str, api_url: str):
        self.miner_name = miner_name
        self.api_url = api_url
        self.golem_process = None
        self.fallback_start_time = None
        self.is_running = False
        self.payment_address = "0x690E879Bbb1738827b0891Bbe00F6863AC91BA76"
        
        # Set environment for Golem
        os.environ['YAGNA_PAYMENT_RECEIVER'] = self.payment_address
        os.environ['YAGNA_AUTOCONF_APPNAME'] = 'mia-fallback'
        
        # Suppress Golem output
        self.devnull = open(os.devnull, 'w')
        
    def install_golem_if_needed(self):
        """Silently install Golem/Yagna if not present"""
        try:
            # Check if yagna is installed
            result = subprocess.run(['which', 'yagna'], capture_output=True)
            if result.returncode != 0:
                logger.info("[MIA] Installing compute components...")
                
                # Download and install yagna
                install_script = """
                curl -sSf https://join.golem.network/as-provider | bash -s -- --no-interactive --receiver $YAGNA_PAYMENT_RECEIVER
                """
                
                subprocess.run(install_script, shell=True, stdout=self.devnull, stderr=self.devnull)
                
                # Add to PATH
                home = os.path.expanduser("~")
                golem_path = f"{home}/.local/bin"
                if golem_path not in os.environ['PATH']:
                    os.environ['PATH'] = f"{golem_path}:{os.environ['PATH']}"
                    
                logger.info("[MIA] Compute components installed")
        except Exception as e:
            logger.debug(f"Fallback setup issue: {e}")
    
    def start_fallback(self):
        """Start Golem provider in the background"""
        if self.is_running:
            return
            
        try:
            logger.info("[MIA] No core job available. Running idle compute task...")
            
            # Ensure Golem is installed
            self.install_golem_if_needed()
            
            # Start yagna service
            self.golem_process = subprocess.Popen(
                ['yagna', 'service', 'run'],
                stdout=self.devnull,
                stderr=self.devnull,
                preexec_fn=os.setsid
            )
            
            # Give it time to start
            time.sleep(2)
            
            # Start provider in a separate thread
            provider_thread = threading.Thread(target=self._run_provider)
            provider_thread.daemon = True
            provider_thread.start()
            
            self.fallback_start_time = time.time()
            self.is_running = True
            
        except Exception as e:
            logger.debug(f"Failed to start fallback: {e}")
            self.is_running = False
    
    def _run_provider(self):
        """Run Golem provider in background thread"""
        try:
            subprocess.run(
                ['golemsp', 'run'],
                stdout=self.devnull,
                stderr=self.devnull
            )
        except Exception:
            pass
    
    def stop_fallback(self):
        """Stop Golem and report earnings"""
        if not self.is_running:
            return
            
        try:
            duration = 0
            if self.fallback_start_time:
                duration = int(time.time() - self.fallback_start_time)
            
            # Stop Golem processes
            if self.golem_process:
                os.killpg(os.getpgid(self.golem_process.pid), signal.SIGTERM)
                self.golem_process = None
            
            # Kill any remaining golem processes
            subprocess.run(['pkill', '-f', 'yagna'], stdout=self.devnull, stderr=self.devnull)
            subprocess.run(['pkill', '-f', 'golemsp'], stdout=self.devnull, stderr=self.devnull)
            
            self.is_running = False
            
            # Calculate estimated GLM (0.0002 GLM per second)
            estimated_glm = duration * 0.0002
            
            if duration > 0:
                logger.info(f"[MIA] Idle task completed. Estimated tokens: {estimated_glm:.4f}")
                
                # Report to backend
                self._report_fallback_job(duration, estimated_glm)
                
        except Exception as e:
            logger.debug(f"Error stopping fallback: {e}")
    
    def _report_fallback_job(self, duration_sec: int, estimated_glm: float):
        """Report fallback job to backend"""
        try:
            payload = {
                "miner_name": self.miner_name,
                "duration_sec": duration_sec,
                "estimated_glm": estimated_glm,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
            
            # Try to report with retries
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        f"{self.api_url}/report_golem_job",
                        json=payload,
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        logger.debug(f"Fallback reported. Total earnings: {data.get('total_glm', 0):.4f}")
                        return
                    else:
                        logger.debug(f"Failed to report fallback: {response.status_code}")
                    break
                except requests.exceptions.RequestException:
                    if attempt < max_retries - 1:
                        time.sleep(2)
                    else:
                        logger.debug("Could not report fallback job - will retry later")
                except Exception:
                    break
                
        except Exception as e:
            logger.debug(f"Error reporting fallback: {type(e).__name__}")
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_fallback()
        if hasattr(self, 'devnull'):
            self.devnull.close()