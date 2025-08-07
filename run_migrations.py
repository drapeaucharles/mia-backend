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

def check_and_add_ip_address_column():
    """Add ip_address column to miners table if missing"""
    try:
        with engine.connect() as conn:
            # Check if column exists
            result = conn.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name='miners' AND column_name='ip_address'
            """))
            
            if result.fetchone():
                print("✓ Column 'ip_address' already exists")
                return True
            
            # Add the column
            print("Adding 'ip_address' column to miners table...")
            conn.execute(text("""
                ALTER TABLE miners 
                ADD COLUMN ip_address VARCHAR(45);
            """))
            conn.commit()
            
            print("✓ Added 'ip_address' column")
            return True
            
    except Exception as e:
        print(f"✗ Failed to add ip_address column: {e}")
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
    check_and_add_ip_address_column()
    
    print("\n✓ All migrations completed successfully!")

if __name__ == "__main__":
    run_all_migrations()