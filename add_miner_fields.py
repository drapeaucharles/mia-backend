"""
Migration script to add missing fields to Miner table
Run this if you get errors about missing columns
"""

from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

load_dotenv()

# Get database URL from environment
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost/mia")

engine = create_engine(DATABASE_URL)

with engine.connect() as conn:
    # Add gpu_info column if it doesn't exist
    try:
        conn.execute(text("""
            ALTER TABLE miners 
            ADD COLUMN IF NOT EXISTS gpu_info TEXT;
        """))
        print("✅ Added gpu_info column")
    except Exception as e:
        print(f"⚠️ gpu_info column might already exist: {e}")
    
    # Add last_seen column if it doesn't exist  
    try:
        conn.execute(text("""
            ALTER TABLE miners 
            ADD COLUMN IF NOT EXISTS last_seen TIMESTAMP;
        """))
        print("✅ Added last_seen column")
    except Exception as e:
        print(f"⚠️ last_seen column might already exist: {e}")
    
    conn.commit()
    print("\n🎉 Migration complete!")