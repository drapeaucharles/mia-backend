#!/usr/bin/env python3
"""
Quick fix for miners table - adds missing ip_address column
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

with engine.connect() as conn:
    try:
        # Try to add the column
        conn.execute(text("ALTER TABLE miners ADD COLUMN IF NOT EXISTS ip_address VARCHAR(45);"))
        conn.commit()
        print("✓ Fixed miners table - added ip_address column")
    except Exception as e:
        # PostgreSQL doesn't support IF NOT EXISTS for columns, try without it
        try:
            conn.execute(text("ALTER TABLE miners ADD COLUMN ip_address VARCHAR(45);"))
            conn.commit()
            print("✓ Fixed miners table - added ip_address column")
        except Exception as e2:
            if "already exists" in str(e2).lower():
                print("✓ Column ip_address already exists")
            else:
                print(f"✗ Error: {e2}")