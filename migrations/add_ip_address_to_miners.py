#!/usr/bin/env python3
"""
Migration script to add ip_address column to miners table
"""

import os
import sys
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

def run_migration():
    """Add ip_address column to miners table if it doesn't exist"""
    
    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        print("ERROR: DATABASE_URL not set")
        return False
    
    engine = create_engine(DATABASE_URL)
    
    try:
        with engine.connect() as conn:
            # Check if column already exists
            result = conn.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name='miners' AND column_name='ip_address'
            """))
            
            if result.fetchone():
                print("✓ Column 'ip_address' already exists in miners table")
                return True
            
            # Add the column
            print("Adding 'ip_address' column to miners table...")
            conn.execute(text("""
                ALTER TABLE miners 
                ADD COLUMN ip_address VARCHAR(45);
            """))
            conn.commit()
            
            print("✓ Successfully added 'ip_address' column to miners table")
            return True
            
    except Exception as e:
        print(f"✗ Migration failed: {e}")
        return False

if __name__ == "__main__":
    success = run_migration()
    sys.exit(0 if success else 1)