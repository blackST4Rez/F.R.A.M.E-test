"""
SQLite3 Database Viewer for Attendance System
This script helps you view and manage the attendance.db database
"""

import sqlite3
import os
from datetime import datetime

DATABASE = 'attendance.db'

def view_tables():
    """View all tables in the database"""
    if not os.path.exists(DATABASE):
        print(f"Database file '{DATABASE}' does not exist yet.")
        print("Run the Flask app first to create the database.")
        return
    
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    
    print("\n" + "="*60)
    print("DATABASE TABLES")
    print("="*60)
    for table in tables:
        print(f"  - {table[0]}")
    
    conn.close()

def view_students():
    """View all students"""
    if not os.path.exists(DATABASE):
        print(f"Database file '{DATABASE}' does not exist yet.")
        return
    
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM student ORDER BY id")
    students = cursor.fetchall()
    
    print("\n" + "="*60)
    print("STUDENTS")
    print("="*60)
    if students:
        print(f"{'ID':<15} {'Name':<20} {'Section':<10} {'Status':<15}")
        print("-"*60)
        for student in students:
            print(f"{student['id']:<15} {student['name']:<20} {student['section'] or 'None':<10} {student['status']:<15}")
    else:
        print("No students found.")
    
    conn.close()

def view_attendance():
    """View all attendance records"""
    if not os.path.exists(DATABASE):
        print(f"Database file '{DATABASE}' does not exist yet.")
        return
    
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT a.id, a.name, a.section, a.time, s.status
        FROM attendance a
        LEFT JOIN student s ON a.id = s.id
        ORDER BY a.time DESC
        LIMIT 50
    """)
    records = cursor.fetchall()
    
    print("\n" + "="*80)
    print("ATTENDANCE RECORDS (Latest 50)")
    print("="*80)
    if records:
        print(f"{'ID':<15} {'Name':<20} {'Section':<10} {'Time':<20} {'Status':<15}")
        print("-"*80)
        for record in records:
            print(f"{record['id']:<15} {record['name']:<20} {record['section'] or 'None':<10} {record['time']:<20} {record['status'] or 'Unknown':<15}")
    else:
        print("No attendance records found.")
    
    conn.close()

def view_admins():
    """View all admin users"""
    if not os.path.exists(DATABASE):
        print(f"Database file '{DATABASE}' does not exist yet.")
        return
    
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM admin_signup")
    admins = cursor.fetchall()
    
    print("\n" + "="*60)
    print("ADMIN USERS")
    print("="*60)
    if admins:
        print(f"{'Admin ID':<20} {'Username':<20}")
        print("-"*60)
        for admin in admins:
            print(f"{admin['admin_id']:<20} {admin['username']:<20}")
    else:
        print("No admin users found.")
    
    conn.close()

def view_admin_logs():
    """View admin login logs"""
    if not os.path.exists(DATABASE):
        print(f"Database file '{DATABASE}' does not exist yet.")
        return
    
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT l.admin_id, s.username, l.login_time
        FROM admin_login l
        JOIN admin_signup s ON l.admin_id = s.admin_id
        ORDER BY l.login_time DESC
        LIMIT 20
    """)
    logs = cursor.fetchall()
    
    print("\n" + "="*80)
    print("ADMIN LOGIN LOGS (Latest 20)")
    print("="*80)
    if logs:
        print(f"{'Admin ID':<20} {'Username':<20} {'Login Time':<30}")
        print("-"*80)
        for log in logs:
            print(f"{log['admin_id']:<20} {log['username']:<20} {log['login_time']:<30}")
    else:
        print("No login logs found.")
    
    conn.close()

def get_database_info():
    """Get database file information"""
    if not os.path.exists(DATABASE):
        print(f"\nDatabase file '{DATABASE}' does not exist yet.")
        print("Location: " + os.path.abspath(DATABASE))
        print("\nRun the Flask app first to create the database.")
        return
    
    file_size = os.path.getsize(DATABASE)
    file_path = os.path.abspath(DATABASE)
    
    print("\n" + "="*60)
    print("DATABASE INFORMATION")
    print("="*60)
    print(f"File Location: {file_path}")
    print(f"File Size: {file_size:,} bytes ({file_size/1024:.2f} KB)")
    
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    # Count records in each table
    tables = ['student', 'attendance', 'admin_signup', 'admin_login']
    print("\nRecord Counts:")
    for table in tables:
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"  {table}: {count} records")
        except:
            print(f"  {table}: table does not exist")
    
    conn.close()

def main():
    """Main menu"""
    print("\n" + "="*60)
    print("SQLite3 Database Viewer - Attendance System")
    print("="*60)
    
    get_database_info()
    
    if os.path.exists(DATABASE):
        view_tables()
        view_students()
        view_attendance()
        view_admins()
        view_admin_logs()
    
    print("\n" + "="*60)
    print("\nTo access the database directly, you can:")
    print("1. Use this script: python view_database.py")
    print("2. Use SQLite3 command line: sqlite3 attendance.db")
    print("3. Use a GUI tool like DB Browser for SQLite")
    print("4. Use Python: import sqlite3; conn = sqlite3.connect('attendance.db')")
    print("="*60 + "\n")

if __name__ == '__main__':
    main()
