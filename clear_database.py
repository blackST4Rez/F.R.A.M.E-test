"""
Clear all data from the database while keeping the database structure intact.
This script deletes all records but preserves the tables.
"""

import sqlite3
import os

DATABASE = 'attendance.db'

def clear_database():
    """Delete all data from all tables while keeping the structure"""
    if not os.path.exists(DATABASE):
        print(f"Database file '{DATABASE}' does not exist.")
        return
    
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        
        # List of tables to clear
        tables = ['attendance', 'admin_login', 'admin_action_log', 'admin_signup', 'student']
        
        print("Clearing database data...")
        print("="*50)
        
        for table in tables:
            cursor.execute(f"DELETE FROM {table}")
            print(f"✓ Cleared '{table}' table")
        
        conn.commit()
        conn.close()
        
        print("="*50)
        print("✓ All database data cleared successfully!")
        print("✓ Database structure and tables remain intact.")
        
    except sqlite3.Error as e:
        print(f"✗ Error clearing database: {e}")

if __name__ == "__main__":
    clear_database()
