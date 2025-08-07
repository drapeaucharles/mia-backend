#!/usr/bin/env python3
"""
Run all database migrations
"""

import os
import sys
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from db import init_db, engine

load_dotenv()

def check_and_add_missing_columns():
    """Add all missing columns to miners table"""
    columns_to_add = [
        ("ip_address", "VARCHAR(45)"),
        ("gpu_name", "VARCHAR(255)"),
        ("gpu_memory_mb", "INTEGER"),
        ("status", "VARCHAR(20) DEFAULT 'idle'"),
        ("job_count", "INTEGER DEFAULT 0"),
        ("total_tokens_generated", "BIGINT DEFAULT 0"),
        ("last_active", "TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP")
    ]
    
    try:
        with engine.connect() as conn:
            for column_name, column_type in columns_to_add:
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
            
            return True
            
    except Exception as e:
        print(f"✗ Failed to add columns: {e}")
        return False

def run_all_migrations():
    """Run all database migrations"""
    print("Running database migrations...")
    
    # Ensure all tables exist
    print("1. Ensuring all tables exist...")
    init_db()
    print("✓ Tables initialized")
    
    # Add missing columns
    print("\n2. Checking for missing columns...")
    check_and_add_missing_columns()
    
    print("\n✓ All migrations completed successfully!")

if __name__ == "__main__":
    run_all_migrations()