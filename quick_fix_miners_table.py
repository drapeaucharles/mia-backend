#!/usr/bin/env python3
"""
Quick fix for miners table - adds all missing columns
Run this on Railway console: python quick_fix_miners_table.py
"""

import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    print("ERROR: DATABASE_URL not set")
    exit(1)

engine = create_engine(DATABASE_URL)

# List of columns to add with their types
columns_to_add = [
    ("ip_address", "VARCHAR(45)"),
    ("gpu_name", "VARCHAR(255)"),
    ("gpu_memory_mb", "INTEGER"),
    ("status", "VARCHAR(20) DEFAULT 'idle'"),
    ("job_count", "INTEGER DEFAULT 0"),
    ("total_tokens_generated", "BIGINT DEFAULT 0"),
    ("last_active", "TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP")
]

with engine.connect() as conn:
    for column_name, column_type in columns_to_add:
        try:
            # Check if column exists
            result = conn.execute(text(f"""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name='miners' AND column_name='{column_name}'
            """))
            
            if result.fetchone():
                print(f"✓ Column '{column_name}' already exists")
            else:
                # Add the column
                conn.execute(text(f"ALTER TABLE miners ADD COLUMN {column_name} {column_type};"))
                conn.commit()
                print(f"✓ Added column '{column_name}'")
                
        except Exception as e:
            if "already exists" in str(e).lower():
                print(f"✓ Column '{column_name}' already exists")
            else:
                print(f"✗ Error adding '{column_name}': {e}")
    
    print("\n✓ Miners table schema fixed!")