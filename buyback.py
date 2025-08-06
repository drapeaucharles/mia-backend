import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session
import db as database
from utils import utc_now

logger = logging.getLogger(__name__)

class BuybackEngine:
    """Manages token buyback and burn operations"""
    
    def __init__(self):
        self.buyback_threshold = float(os.getenv("BUYBACK_THRESHOLD_USD", "100.0"))
        self.token_symbol = os.getenv("TOKEN_SYMBOL", "SERV")
        self.burn_address = os.getenv("BURN_ADDRESS", "0x000000000000000000000000000000000000dEaD")
        
        # Simulated exchange rates (in production, fetch from real sources)
        self.usd_per_token = 0.10  # $0.10 per SERV token
    
    def get_income_balance(self, db: Session) -> float:
        """Get current RunPod income balance"""
        metric = db.query(database.SystemMetrics).filter(
            database.SystemMetrics.metric_name == "runpod_income_usd"
        ).first()
        
        return metric.value if metric else 0.0
    
    def update_income_balance(self, db: Session, new_balance: float):
        """Update RunPod income balance"""
        metric = db.query(database.SystemMetrics).filter(
            database.SystemMetrics.metric_name == "runpod_income_usd"
        ).first()
        
        if metric:
            metric.value = new_balance
            metric.updated_at = utc_now()
        else:
            metric = database.SystemMetrics(
                metric_name="runpod_income_usd",
                value=new_balance
            )
            db.add(metric)
        
        db.commit()
    
    def update_buyback_metrics(self, db: Session, amount_usd: float):
        """Update buyback-related metrics"""
        # Update total buyback amount
        total_metric = db.query(database.SystemMetrics).filter(
            database.SystemMetrics.metric_name == "total_buyback_usd"
        ).first()
        
        if total_metric:
            total_metric.value += amount_usd
        else:
            total_metric = database.SystemMetrics(
                metric_name="total_buyback_usd",
                value=amount_usd
            )
            db.add(total_metric)
        
        # Update last buyback timestamp
        timestamp_metric = db.query(database.SystemMetrics).filter(
            database.SystemMetrics.metric_name == "last_buyback_timestamp"
        ).first()
        
        if timestamp_metric:
            timestamp_metric.value = datetime.utcnow().timestamp()
        else:
            timestamp_metric = database.SystemMetrics(
                metric_name="last_buyback_timestamp",
                value=datetime.utcnow().timestamp()
            )
            db.add(timestamp_metric)
        
        db.commit()
    
    def calculate_tokens_to_buy(self, usd_amount: float) -> float:
        """Calculate how many tokens can be bought with USD amount"""
        # In production, this would fetch real market prices
        # Add slippage and fees consideration
        slippage = 0.02  # 2% slippage
        fees = 0.003  # 0.3% trading fees
        
        effective_amount = usd_amount * (1 - slippage - fees)
        tokens = effective_amount / self.usd_per_token
        
        return round(tokens, 2)
    
    def simulate_market_buy(self, usd_amount: float) -> Dict[str, Any]:
        """Simulate buying tokens from the market"""
        tokens_bought = self.calculate_tokens_to_buy(usd_amount)
        
        # In production, this would interact with a DEX or CEX API
        return {
            "success": True,
            "usd_spent": usd_amount,
            "tokens_bought": tokens_bought,
            "price_per_token": self.usd_per_token,
            "exchange": "Simulated DEX",
            "transaction_hash": f"0x{''.join(['abcdef0123456789'[i % 16] for i in range(64)])}"
        }
    
    def simulate_token_burn(self, token_amount: float) -> Dict[str, Any]:
        """Simulate burning tokens"""
        # In production, this would send tokens to burn address
        return {
            "success": True,
            "tokens_burned": token_amount,
            "burn_address": self.burn_address,
            "transaction_hash": f"0x{''.join(['fedcba9876543210'[i % 16] for i in range(64)])}"
        }
    
    def check_and_execute_buyback(self, db: Session) -> Dict[str, Any]:
        """Check if buyback should be triggered and execute if needed"""
        current_balance = self.get_income_balance(db)
        
        if current_balance < self.buyback_threshold:
            return {
                "triggered": False,
                "reason": "Income below threshold",
                "current_balance": current_balance,
                "threshold": self.buyback_threshold
            }
        
        # Execute buyback
        try:
            # Simulate market buy
            buy_result = self.simulate_market_buy(current_balance)
            
            if not buy_result["success"]:
                return {
                    "triggered": False,
                    "reason": "Market buy failed",
                    "error": buy_result.get("error")
                }
            
            # Simulate token burn
            burn_result = self.simulate_token_burn(buy_result["tokens_bought"])
            
            if not burn_result["success"]:
                return {
                    "triggered": False,
                    "reason": "Token burn failed",
                    "error": burn_result.get("error")
                }
            
            # Update balances and metrics
            self.update_income_balance(db, 0.0)  # Reset income balance
            self.update_buyback_metrics(db, current_balance)
            
            logger.info(f"Buyback executed: ${current_balance} -> {buy_result['tokens_bought']} {self.token_symbol} burned")
            
            return {
                "triggered": True,
                "amount_usd": current_balance,
                "tokens_bought": buy_result["tokens_bought"],
                "tokens_burned": burn_result["tokens_burned"],
                "buy_tx_hash": buy_result["transaction_hash"],
                "burn_tx_hash": burn_result["transaction_hash"],
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Buyback execution error: {e}")
            return {
                "triggered": False,
                "reason": "Execution error",
                "error": str(e)
            }
    
    def get_buyback_history(self, db: Session) -> Dict[str, Any]:
        """Get buyback history and statistics"""
        metrics = {}
        
        # Get all relevant metrics
        metric_names = ["total_buyback_usd", "last_buyback_timestamp", "runpod_income_usd"]
        
        for metric_name in metric_names:
            metric = db.query(database.SystemMetrics).filter(
                database.SystemMetrics.metric_name == metric_name
            ).first()
            
            if metric:
                if metric_name == "last_buyback_timestamp" and metric.value > 0:
                    metrics[metric_name] = datetime.fromtimestamp(metric.value).isoformat()
                else:
                    metrics[metric_name] = metric.value
            else:
                metrics[metric_name] = None if "timestamp" in metric_name else 0.0
        
        # Calculate estimated tokens burned
        if metrics["total_buyback_usd"] > 0:
            metrics["estimated_tokens_burned"] = self.calculate_tokens_to_buy(metrics["total_buyback_usd"])
        else:
            metrics["estimated_tokens_burned"] = 0.0
        
        metrics["buyback_threshold"] = self.buyback_threshold
        metrics["next_buyback_in_usd"] = max(0, self.buyback_threshold - metrics.get("runpod_income_usd", 0))
        
        return metrics