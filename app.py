import os
import csv
import cv2
import shutil
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from flask import Flask,render_template,redirect,request,session,g,url_for,make_response
from datetime import datetime, date
import joblib
from dotenv import load_dotenv
import tensorflow as tf
from tensorflow import keras
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import pickle
from flask_talisman import Talisman
from flask_wtf.csrf import CSRFProtect
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

load_dotenv()
# Initializing the flask app

app=Flask(__name__)

# Security: Content Security Policy (CSP)
# Since we use inline scripts and styles, and external CDNs, we need to configure CSP carefully
csp = {
    'default-src': ["'self'", 'https://cdn.jsdelivr.net', 'https://fonts.googleapis.com', 'https://fonts.gstatic.com'],
    'script-src': ["'self'", "'unsafe-inline'", 'https://cdn.jsdelivr.net'],
    'style-src': ["'self'", "'unsafe-inline'", 'https://fonts.googleapis.com', 'https://cdn.jsdelivr.net'],
    'img-src': ["'self'", 'data:'],
    'font-src': ["'self'", 'https://fonts.gstatic.com']
}

# Force HTTPS in production, but allow HTTP for local testing
Talisman(app, content_security_policy=csp, force_https=False, session_cookie_secure=False)

# Security: CSRF Protection
csrf = CSRFProtect(app)

# Security: Rate Limiting
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

# Set secret key from environment variable, or use a fallback for development
app.secret_key = os.getenv('SECRET_KEY') or 'dev-secret-key-change-in-production-12a6bfc9462626cc3ccbf316185931465fb8b1041699bb4f8cfc10418305280c'

# Security: Session Cookie Configuration
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    SESSION_COOKIE_SECURE=False, # Allow cookies over HTTP for LAN access
    SESSION_COOKIE_PATH='/',
    SESSION_COOKIE_DOMAIN=None, # Allow all domains/IPs
)

# Flask error handler
@app.errorhandler(404)
@app.errorhandler(401)
@app.errorhandler(500)
def http_error_handler(error):
    return render_template('Error.html')

# SQLite3 database configuration
DATABASE = 'attendance.db'

# Database connection helper
def get_db():
    """Get SQLite3 database connection with row factory for dict-like access"""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row  # This makes rows accessible like dictionaries
    return conn

# Initialize database tables
def init_db():
    """Create database tables if they don't exist"""
    conn = get_db()
    cur = conn.cursor()
    
    # Create student table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS student (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            section TEXT,
            status TEXT NOT NULL
        )
    """)
    
    # Create attendance table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id TEXT NOT NULL,
            name TEXT NOT NULL,
            section TEXT,
            time TEXT NOT NULL,
            FOREIGN KEY (id) REFERENCES student(id)
        )
    """)
    
    # Create admin_signup table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS admin_signup (
            admin_id TEXT PRIMARY KEY,
            username TEXT NOT NULL,
            password TEXT NOT NULL
        )
    """)
    
    # Create admin_login table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS admin_login (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            admin_id TEXT NOT NULL,
            username TEXT NOT NULL,
            login_time TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (admin_id) REFERENCES admin_signup(admin_id)
        )
    """)
    
    # Create admin_action_log table for detailed audit trail
    cur.execute("""
        CREATE TABLE IF NOT EXISTS admin_action_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            admin_id TEXT NOT NULL,
            action TEXT NOT NULL,
            details TEXT,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (admin_id) REFERENCES admin_signup(admin_id)
        )
    """)
    
    conn.commit()
    conn.close()

# Helper function to log admin actions
def log_admin_action(admin_id, action, details):
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("INSERT INTO admin_action_log (admin_id, action, details) VALUES (?, ?, ?)", 
                   (admin_id, action, details))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error logging action: {e}")

    
# Flask assign admin
@app.before_request
def before_request():
    g.user = None
    # Debug cookies
    # print(f"[DEBUG] Cookies: {request.cookies}")
    
    if 'admin' in session:
        # Debug print
        print(f"[DEBUG] Session admin found: {session['admin']}")
        # Verify if admin still exists in database
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM admin_signup WHERE admin_id = ?", (session['admin'],))
        admin_exists = cur.fetchone()
        conn.close()
        
        if admin_exists:
            g.user = session['admin']
        else:
            # Admin account was deleted - log them out immediately
            session.pop('admin', None)
            # Redirect to home page (skip for static files to avoid breaking page assets)
            if request.endpoint and 'static' not in request.endpoint:
                return redirect(url_for('home'))

# Current Date & Time
datetoday = date.today().strftime("%d-%m-%Y")
datetoday2 = date.today().strftime("%d %B %Y")

# Capture the video
base_dir = os.path.dirname(os.path.abspath(__file__))
face_detector_path = os.path.join(base_dir, 'static', 'haarcascade_frontalface_default.xml')
eye_detector_path = os.path.join(base_dir, 'static', 'haarcascade_eye.xml')

face_detector = cv2.CascadeClassifier(face_detector_path)
eye_detector = cv2.CascadeClassifier(eye_detector_path)

if face_detector.empty():
    print(f"[Error] Could not load face detector from {face_detector_path}")
if eye_detector.empty():
    print(f"[Error] Could not load eye detector from {eye_detector_path}")

cap = cv2.VideoCapture(0)

# ======= Check and Make Folders ========
folders = [
    os.path.join(base_dir, 'static', 'faces'),
    os.path.join(base_dir, 'final_model')
]
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    
# ======= Global Variables =======
cnn_model = None
class_names = {}
        
# ======= Total Registered Users ========
def totalreg():
    faces_dir = os.path.join(base_dir, 'static', 'faces')
    if not os.path.isdir(faces_dir):
        return 0
    return len([name for name in os.listdir(faces_dir) if os.path.isdir(os.path.join(faces_dir, name))])

# ======= Get Face From Image =========
def extract_faces_and_eyes(img):
    if img is None or img.size == 0:
        return (), ()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    eyes_list = []

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_detector.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3)
        eyes_global = [(ex + x, ey + y, ew, eh) for (ex, ey, ew, eh) in eyes]
        eyes_list.append(eyes_global)

    return faces, eyes_list

# ======= Identify Face Using CNN Model ========
def identify_face(face_img):
    global cnn_model, class_names
    
    if face_img is None or cnn_model is None:
        print("[DEBUG] No model or image available")
        return "Unknown"

    try:
        # Preprocess face image
        face_resized = cv2.resize(face_img, (224, 224))
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        face_normalized = face_rgb.astype('float32') / 255.0
        face_input = np.expand_dims(face_normalized, axis=0)

        # Predict and get class with confidence
        preds = cnn_model.predict(face_input, verbose=0)[0]
        top1 = int(np.argmax(preds))
        top2 = int(np.argsort(preds)[-2]) if preds.size >= 2 else top1
        p1 = float(preds[top1])
        p2 = float(preds[top2])

        # Confidence floor and separation margin vs runner-up to avoid near-tie flips
        CONFIDENCE_THRESHOLD = 0.5
        MARGIN = 0.1

        if p1 < CONFIDENCE_THRESHOLD:
            return "Unknown"
            
        # Map to class name
        return class_names.get(top1, "Unknown")
    
    except Exception as e:
        print(f"[Error] Face identification failed: {e}")
        return "Unknown"

# ======= Train Model Using Available Faces ========
def train_model():
    global cnn_model, class_names

    face_dir = os.path.join(base_dir, 'static', 'faces')
    model_path = os.path.join(base_dir, 'final_model', 'face_recognition_model.h5')
    class_path = os.path.join(base_dir, 'final_model', 'class_names.pkl')

    if not os.path.exists(face_dir) or len(os.listdir(face_dir)) == 0:
        print("[Info] No faces to train on.")
        return
        
    # Remove old model if exists
    if os.path.exists(model_path):
        os.remove(model_path)
    if os.path.exists(class_path):
        os.remove(class_path)

    # Data augmentation
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.9, 1.1],
        validation_split=0.2
    )

    # Train and validation generators
    train_data = datagen.flow_from_directory(
        face_dir, target_size=(224,224), batch_size=32,
        class_mode='categorical', subset='training', shuffle=True
    )
    val_data = datagen.flow_from_directory(
        face_dir, target_size=(224,224), batch_size=32,
        class_mode='categorical', subset='validation', shuffle=True
    )
    # Build CNN model
    cnn_model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(224,224,3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(128, (3,3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(train_data.num_classes, activation="softmax")
    ])

    cnn_model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics=["accuracy"]
    )

    # Train the model
    print("[INFO] Training model...")
    cnn_model.fit(
        train_data, steps_per_epoch=len(train_data),
        validation_data=val_data, validation_steps=len(val_data),
        epochs=20, verbose=1
    )

    # Save model and class names
    os.makedirs(os.path.join(base_dir, 'final_model'), exist_ok=True)
    cnn_model.save(model_path)
    
    # Create class names mapping from training data
    class_names = {v: k for k, v in train_data.class_indices.items()}
    with open(class_path, 'wb') as f:
        pickle.dump(class_names, f)

    print("Model trained and saved successfully!")
    print(f"Classes: {class_names}")
    
# ======= Load CNN Model =======
def load_cnn_model():
    global cnn_model, class_names

    model_path = os.path.join(base_dir, 'final_model', 'face_recognition_model.h5')
    class_names_path = os.path.join(base_dir, 'final_model', 'class_names.pkl')

    # Load CNN model
    if os.path.exists(model_path):
        try:
            cnn_model = tf.keras.models.load_model(model_path)
        except Exception as e:
            cnn_model = None
    else:
        print(f"[WARNING] CNN model not found at {model_path}.")
        cnn_model = None

    # Load class_names
    if os.path.exists(class_names_path):
        try:
            with open(class_names_path, 'rb') as f:
                class_names = pickle.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load class names: {e}")
            class_names = {}
    else:
        print(f"[WARNING] class_names.pkl not found.")
        class_names = {}

    # Validate saved classes vs current face folders to avoid stale labels influencing predictions
    try:
        faces_root = os.path.join(base_dir, 'static', 'faces')
        if os.path.isdir(faces_root):
            current_dirs = {d for d in os.listdir(faces_root) if os.path.isdir(os.path.join(faces_root, d))}
            saved_labels = set(class_names.values()) if isinstance(class_names, dict) else set()
            if saved_labels and not saved_labels.issubset(current_dirs):
                print("[INFO] Detected class mismatch with current folders. Retraining model...")
                train_model()
                if os.path.exists(model_path):
                    cnn_model = tf.keras.models.load_model(model_path)
                if os.path.exists(class_names_path):
                    with open(class_names_path, 'rb') as f:
                        class_names = pickle.load(f)
    except Exception as e:
        print(f"[WARN] Class validation failed: {e}")

# ======= Remove Attendance of Deleted User ======
def remAttendance():
    conn = get_db()
    cur = conn.cursor()

    # Collect valid IDs from both user tables
    cur.execute("SELECT id FROM student WHERE status='registered'")
    registered_ids = {str(row['id']) for row in cur.fetchall()}

    cur.execute("SELECT id FROM student WHERE status='unregistered'")
    unregistered_ids = {str(row['id']) for row in cur.fetchall()}

    valid_ids = registered_ids | unregistered_ids

    # If there are valid IDs, remove all attendance records that don't belong to them
    if valid_ids:
        placeholders = ','.join(['?' for _ in valid_ids])
        cur.execute(f"DELETE FROM attendance WHERE id NOT IN ({placeholders})", list(valid_ids))

    conn.commit()
    conn.close()

# ======== Get Info From Attendance File =========
def extract_attendance():
    conn = get_db()
    cur = conn.cursor()

    datetoday_sqlite = date.today().strftime("%Y-%m-%d")
        
    query = """
        SELECT a.name, a.id, a.section, a.time,
               COALESCE(s.status, 'Unknown') AS status
        FROM attendance a
        LEFT JOIN student s ON a.id = s.id
        WHERE date(a.time) = ?
        ORDER BY a.time ASC
    """
    cur.execute(query, (datetoday_sqlite,))
    rows = cur.fetchall()
    conn.close()

    if not rows:
        return [], [], [], [], [], [], 0

    names = [r['name'] for r in rows]
    rolls = [r['id'] for r in rows]
    sec   = [r['section'] for r in rows]
    # SQLite3 stores time as string, so parse it
    times = []
    for r in rows:
        time_str = r['time']
        if isinstance(time_str, str) and ' ' in time_str:
            times.append(time_str.split(' ')[1][:8])  # Extract time part
        else:
            times.append(str(time_str)[:8])
    reg   = [r['status'] for r in rows]
    l     = len(rows)
    datetoday_disp = date.today().strftime("%d-%m-%Y")
    dates = [datetoday_disp] * l

    return names, rolls, sec, times, dates, reg, l

# ======== Save Attendance =========
def add_attendance(name):
    username, userid, usersection = name.split('$')
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = get_db()
    cur = conn.cursor()

    # Check if already marked today (ignoring time, only by DATE)
    cur.execute("""
        SELECT * FROM attendance 
        WHERE id=? AND date(time)=?
    """, (userid, datetime.now().strftime("%Y-%m-%d")))
    already = cur.fetchone()

    if already:
        conn.close()
        return 

    # Insert new attendance with full date+time
    cur.execute("""
        INSERT INTO attendance (id, name, section, time)
        VALUES (?, ?, ?, ?)
    """, (userid, username, usersection, current_datetime))
    conn.commit()
    conn.close()

# ======= Flask Home Page =========
@app.route('/')
def home():
    resp = None
    if g.user:
        resp = make_response(render_template('HomePage.html', admin=True, mess='Logged in as Administrator', user=session['admin']))
    else:
        resp = make_response(render_template('HomePage.html', admin=False, datetoday2=datetoday2))
    
    # Disable caching
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp

@app.route('/attendance')
def take_attendance():
    # Fetch today's attendance from MySQL
    names, rolls, sec, times, dates, reg, l = extract_attendance()
    
    return render_template(
        'Attendance.html',
        names=names,
        rolls=rolls,
        sec=sec,
        times=times,
        l=l,
        datetoday2=datetoday2
    )
    
    
@app.route('/attendancebtn', methods=['GET'])
def attendancebtn():
    global cnn_model, class_names
    
    faces_dir = os.path.join(base_dir, 'static', 'faces')
    # Check if faces directory exists and is not empty
    if not os.path.exists(faces_dir):
        os.makedirs(faces_dir, exist_ok=True)
        
    faces_count = len([name for name in os.listdir(faces_dir) if os.path.isdir(os.path.join(faces_dir, name))])
    
    if faces_count == 0:
        names, rolls, sec, times, dates, reg, l = extract_attendance()
        return render_template('Attendance.html', datetoday2=datetoday2,
                               names=names, rolls=rolls, sec=sec, times=times, l=l,
                               mess='Database is empty! Register yourself first.')

    # Ensure model exists and is loaded
    if cnn_model is None:
        print("[INFO] Model not loaded, attempting to load...")
        load_cnn_model()
        
    if cnn_model is None:
        print("[INFO] No model found, training new model...")
        train_model()
        load_cnn_model()

    if cnn_model is None:
        names, rolls, sec, times, dates, reg, l = extract_attendance()
        return render_template('Attendance.html', datetoday2=datetoday2,
                               names=names, rolls=rolls, sec=sec, times=times, l=l,
                               mess='Failed to load or train model.')

    cap = cv2.VideoCapture(0)
    if cap is None or not cap.isOpened():
        names, rolls, sec, times, dates, reg, l = extract_attendance()
        return render_template('Attendance.html', names=names, rolls=rolls, sec=sec, times=times, l=l,
                               totalreg=totalreg(), datetoday2=datetoday2, mess='Camera not available.')

    ret = True
    # temporal smoothing & lock-on
    consecutive_counts = {}
    current_lock = None
    lock_grace = 10  # frames to keep lock if momentarily lost
    lock_timer = 0
    NEED_CONSEC = 5   # frames required to confirm identity

    while ret:
        ret, frame = cap.read()
        faces, eyes_list = extract_faces_and_eyes(frame)

        identified_person_name = "Unknown"
        identified_person_id = "N/A"

        if faces is not None and len(faces) > 0:
            for i, (x, y, w, h) in enumerate(faces):
                if w < 100 or h < 100:  # Skip very small faces
                    continue
                    
                if i < len(eyes_list) and len(eyes_list[i]) > 0:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
                    for (ex, ey, ew, eh) in eyes_list[i]:
                        cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

                    face_img = cv2.resize(frame[y:y + h, x:x + w], (224, 224))
                    identified_person = identify_face(face_img)

                    # If we already locked an identity, keep it as long as lock_timer remains
                    if current_lock is not None and '$' in current_lock:
                        identified_person_name, identified_person_id, *_ = current_lock.split('$')
                        lock_timer = lock_grace
                    else:
                        # Build up consecutive evidence before locking and marking attendance
                        if identified_person is not None and '$' in identified_person:
                            consecutive_counts[identified_person] = consecutive_counts.get(identified_person, 0) + 1
                            if consecutive_counts[identified_person] >= NEED_CONSEC:
                                current_lock = identified_person
                                consecutive_counts.clear()
                                add_attendance(current_lock)
                                identified_person_name, identified_person_id, *_ = current_lock.split('$')
                        else:
                            consecutive_counts.clear()

                    cv2.putText(frame, f'Name: {identified_person_name}', (30, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
                    cv2.putText(frame, f'ID: {identified_person_id}', (30, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
                    cv2.putText(frame, 'Press Esc to close', (30, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 127, 255), 2, cv2.LINE_AA)

        else:
            # if no faces, decay the lock
            if current_lock is not None:
                lock_timer -= 1
                if lock_timer <= 0:
                    current_lock = None

        cv2.namedWindow('Attendance', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Attendance', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('Attendance', frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    names, rolls, sec, times, dates, reg, l = extract_attendance()
    return render_template('Attendance.html', names=names, rolls=rolls, sec=sec, times=times, l=l,
                           datetoday2=datetoday2)

@app.route('/adduser')
def add_user():
    return render_template('AddUser.html')

@app.route('/adduserbtn', methods=['GET', 'POST'])
def adduserbtn():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    newusersection = request.form['newusersection']

    # Open camera
    cap = cv2.VideoCapture(0)
    if cap is None or not cap.isOpened():
        return render_template('AddUser.html', mess='Camera not available.')

    # Create user folder for storing images
    userimagefolder = os.path.join(base_dir, 'static', 'faces', f'{newusername}${newuserid}${newusersection}')
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)

    # Check if user already exists in DB
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM student WHERE id = ?", (newuserid,))
    existing_user = cur.fetchone()
    conn.close()  # Close connection regardless of result

    if existing_user:
        cap.release()
        return render_template('AddUser.html', mess='User already exists in database.')

    images_captured = 0
    max_images = 100

    while images_captured < max_images:
        ret, frame = cap.read()
        if not ret:
            break

        faces, eyes_list = extract_faces_and_eyes(frame)
        if faces is not None:
            for i, (x, y, w, h) in enumerate(faces):
                if i < len(eyes_list) and len(eyes_list[i]) > 0:
                    face_img = cv2.resize(frame[y:y+h, x:x+w], (224,224))
                    cv2.imwrite(
                        os.path.join(userimagefolder, f'{images_captured}.jpg'),
                        face_img
                    )
                    images_captured += 1

                    cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,20), 2)
                    for (ex, ey, ew, eh) in eyes_list[i]:
                        cv2.rectangle(frame, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)

        cv2.putText(frame, f'Images Captured: {images_captured}/{max_images}', (30,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,20), 2)
        cv2.namedWindow("Collecting Face Data", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Collecting Face Data", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Collecting Face Data", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    if images_captured == 0:
        shutil.rmtree(userimagefolder)
        return render_template('AddUser.html', mess='Failed to capture valid face images.')

    # Insert new user into SQLite
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO student (name, id, section, status)
            VALUES (?, ?, ?, 'unregistered')
        """, (newusername, newuserid, newusersection))
        conn.commit()
        conn.close()
    except sqlite3.IntegrityError:
        conn.close()
        # Clean up the captured images if DB insert fails
        if os.path.exists(userimagefolder):
            shutil.rmtree(userimagefolder)
        return render_template('AddUser.html', mess='User ID already exists. Please use a unique ID.')
    except Exception as e:
        conn.close()
        print(f"[Error] Database insertion failed: {e}")
        return render_template('AddUser.html', mess='An error occurred while saving user data.')

    # Retrain model immediately with new user
    train_model()
    load_cnn_model()  # Reload the model after training

    # Fetch updated unregistered students count
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM student WHERE status='unregistered'")
    count = cur.fetchone()[0]
    conn.close()

    return render_template('HomePage.html', admin=False, datetoday2=datetoday2,
                           mess=f'Registration successful! You are now pending approval. (Queue: {count})')

@app.route('/attendancelist')
def attendance_list():
    if not g.user:
        return render_template('LogInForm.html')

    names, rolls, sec, times, dates, reg, l = extract_attendance()
    return render_template('AttendanceList.html', names=names, rolls=rolls, sec=sec, times=times, dates=dates, reg=reg,
                           l=l)
    
# ========== Flask Search Attendance by Date ============
@app.route('/attendancelistdate', methods=['GET', 'POST'])
def attendancelistdate():
    if not g.user:
        return render_template('LogInForm.html')

    date_selected = request.form['date']  # "YYYY-MM-DD"

    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT a.name, a.id, a.section, a.time,
               COALESCE(s.status, 'Unknown') AS status
        FROM attendance a
        LEFT JOIN student s ON a.id = s.id
        WHERE date(a.time) = ?
        ORDER BY a.time ASC
    """, (date_selected,))
    rows = cur.fetchall()
    conn.close()

    if not rows:
        return render_template('AttendanceList.html', names=[], rolls=[], sec=[], times=[], reg=[], l=0,
                               mess="No records for this date.")

    names = [r['name'] for r in rows]
    rolls = [r['id'] for r in rows]
    sec = [r['section'] for r in rows]
    # SQLite3 stores time as string, parse it
    times = []
    dates = []
    for r in rows:
        time_str = r['time']
        if isinstance(time_str, str) and ' ' in time_str:
            times.append(time_str.split(' ')[1][:8])  # Extract time part
            dates.append(time_str.split(' ')[0])  # Extract date part
        else:
            times.append(str(time_str)[:8])
            dates.append(str(time_str)[:10])
    reg = [r['status'] for r in rows]
    l = len(rows)

    return render_template('AttendanceList.html',
                        names=names, rolls=rolls, sec=sec,
                        times=times, dates=dates, reg=reg,
                        l=l, mess=f"Total Attendance: {l}")

# ========== Flask Search Attendance by ID ============
@app.route('/attendancelistid', methods=['GET', 'POST'])
def attendancelistid():
    if not g.user:
        return render_template('LogInForm.html')

    student_id = request.form.get('id')
    if not student_id:
        return render_template('AttendanceList.html', names=[], rolls=[], sec=[], times=[], dates=[], reg=[], l=0, mess="No ID provided!")

    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM attendance WHERE id = ?", (student_id,))
    rows = cur.fetchall()
    conn.close()

    if rows:
        names = [row['name'] for row in rows]
        rolls = [row['id'] for row in rows]
        sec   = [row['section'] for row in rows]
        # SQLite3 stores time as string, parse it
        times = []
        dates = []
        for row in rows:
            time_str = row['time']
            if time_str and isinstance(time_str, str) and ' ' in time_str:
                times.append(time_str.split(' ')[1][:8])  # Extract time part
                dates.append(time_str.split(' ')[0])  # Extract date part
            else:
                times.append("N/A")
                dates.append("N/A")
        reg   = ['Registered' if row['id'] in [r['id'] for r in rows] else 'Unregistered' for row in rows]
        l = len(rows)
        return render_template('AttendanceList.html',
                               names=names, rolls=rolls, sec=sec,
                               times=times, dates=dates, reg=reg,
                               l=l, mess=f"Total Attendance: {l}")
    else:
        return render_template('AttendanceList.html',
                               names=[], rolls=[], sec=[],
                               times=[], dates=[], reg=[],
                               l=0, mess="Nothing Found!")

# ========== Flask Unregister a User ============
@app.route('/unregisteruser', methods=['POST'])
def unregisteruser():
    if not g.user:
        return render_template('LogInForm.html')

    try:
        idx = int(request.form['index'])
    except (ValueError, KeyError):
        return "Invalid index (not a number or missing)", 400

    conn = get_db()
    cur = conn.cursor()
    # Get only registered students
    cur.execute("SELECT * FROM student WHERE status='registered' ORDER BY id ASC")
    registered = cur.fetchall()

    if idx < 0 or idx >= len(registered):
        conn.close()
        return "Invalid index", 400

    user = registered[idx]
    userid, username, section = user['id'], user['name'], user['section']

        # Move the face folder (optional)
    old_folder = f"static/faces/{username}${userid}${section}"
    new_folder = f"static/faces/{username}${userid}$None"
    if os.path.exists(old_folder):
        if os.path.exists(new_folder):
            shutil.rmtree(new_folder)
        shutil.move(old_folder, new_folder)
        
    # Update status in single student table
    cur.execute(
        "UPDATE student SET status='unregistered', section=NULL WHERE id=?",
        (userid,)
    )
    conn.commit()
    
    # Log action
    log_admin_action(session['admin'], 'UNREGISTER USER', f'Unregistered user {userid} ({username})')

    # Return updated list of registered students
    cur.execute("SELECT * FROM student WHERE status='registered' ORDER BY id ASC")
    rows = cur.fetchall()
    conn.close()

    names = [r['name'] for r in rows]
    rolls = [r['id'] for r in rows]
    sec = [r['section'] for r in rows]
    l = len(rows)

    mess = f'Number of Registered Students: {l}' if l > 0 else "Database is empty!"

    return render_template('RegisterUserList.html', names=names, rolls=rolls, sec=sec, l=l, mess=mess)

# ========== Flask Unregister User List ============
@app.route('/unregisteruserlist')
def unregister_user_list():
    if not g.user:
        return render_template('LogInForm.html')
    
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM student WHERE status='unregistered' ORDER BY id ASC")
    rows = cur.fetchall()
    conn.close()

    if not rows:
        return render_template('UnregisterUserList.html', names=[], rolls=[], sec=[], l=0, mess="Database is empty!")

    names = [row['name'] for row in rows]
    rolls = [row['id'] for row in rows]
    sec = [row['section'] for row in rows]
    l = len(rows)

    return render_template('UnregisterUserList.html', names=names, rolls=rolls, sec=sec, l=l,
                           mess=f'Number of Unregistered Students: {l}')

# ========== Flask Delete a User from Unregistered List ============
@app.route('/deleteunregistereduser', methods=['POST'])
def deleteunregistereduser():
    if not g.user:
        return render_template('LogInForm.html')

    idx = int(request.form['index'])

    conn = get_db()
    cur = conn.cursor()
    # Fetch unregistered students only
    cur.execute("SELECT * FROM student WHERE status='unregistered' ORDER BY id ASC")
    unregistered = cur.fetchall()

    if idx >= len(unregistered):
        conn.close()
        return render_template('UnregisterUserList.html', names=[], rolls=[], sec=[], l=0, mess="Invalid user index.")

    user = unregistered[idx]
    username, userid, usersec = user['name'], user['id'], user['section']

    folder = f'static/faces/{username}${userid}${usersec}'
    if os.path.exists(folder):
        shutil.rmtree(folder)
        train_model()

    # Delete student from table
    cur.execute("DELETE FROM student WHERE id = ? AND status='unregistered'", (userid,))
    conn.commit()
    conn.close()
    
    # Log action
    log_admin_action(session['admin'], 'DELETE USER', f'Deleted unregistered user {userid} ({username})')

    return redirect(url_for('unregister_user_list'))
    
# ========== Flask Register User List ============
@app.route('/registeruserlist')
def register_user_list():
    if not g.user:
        return render_template('LogInForm.html')

    conn = get_db()
    cur = conn.cursor()

    cur.execute("SELECT * FROM student WHERE status='registered' ORDER BY id ASC")
    rows = cur.fetchall()
    conn.close()

    names = [row['name'] for row in rows]
    rolls = [row['id'] for row in rows]
    sec = [row['section'] for row in rows]
    l = len(rows)

    mess = f'Number of Registered Students: {l}' if l else "Database is empty!"
    return render_template('RegisterUserList.html', names=names, rolls=rolls, sec=sec, l=l, mess=mess)
        
# ========== Flask Register a User ============
@app.route('/registeruser', methods=['POST'])
def registeruser():
    if not g.user:
        return render_template('LogInForm.html')

    try:
        idx = int(request.form['index'])
        section = request.form['section']
    except (ValueError, KeyError):
        return "Invalid input", 400

    conn = get_db()
    cur = conn.cursor()
    # Get all unregistered students
    cur.execute("SELECT * FROM student WHERE status='unregistered' ORDER BY id ASC")
    unregistered = cur.fetchall()

    if idx < 0 or idx >= len(unregistered):
        conn.close()
        return "Invalid user index", 400

    user = unregistered[idx]
    name, userid = user['name'], user['id']

    # Move the face folder
    old_folder = f"static/faces/{name}${userid}$None"
    new_folder = f"static/faces/{name}${userid}${section}"
    if os.path.exists(old_folder):
        if os.path.exists(new_folder):
            shutil.rmtree(new_folder)
        shutil.move(old_folder, new_folder)

    # Update status and section in single student table
    cur.execute(
        "UPDATE student SET status='registered', section=? WHERE id=?",
        (section, userid)
    )
    conn.commit()

    # Log action
    log_admin_action(session['admin'], 'APPROVE USER', f'Approved user {userid} ({name}) to section {section}')

    # Reload unregistered list
    cur.execute("SELECT * FROM student WHERE status='unregistered' ORDER BY id ASC")
    rows = cur.fetchall()
    conn.close()

    names = [r['name'] for r in rows]
    rolls = [r['id'] for r in rows]  # Fixed: use 'id' instead of 'user_id'
    secs = [r['section'] for r in rows]
    l = len(rows)

    mess = f'Number of Unregistered Students: {l}' if l > 0 else "Database is empty!"
    return render_template('UnregisterUserList.html', names=names, rolls=rolls, sec=secs, l=l, mess=mess)
        
# ========== Flask Delete a User from Registered List ============
@app.route('/deleteregistereduser', methods=['POST'])
def deleteregistereduser():
    if not g.user:
        return render_template('LogInForm.html')

    idx = int(request.form['index'])

    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM student WHERE status='registered' ORDER BY id ASC")
    registered = cur.fetchall()

    if idx >= len(registered):
        conn.close()
        return render_template('RegisterUserList.html', names=[], rolls=[], sec=[], l=0, mess="Invalid user index.")

    user = registered[idx]
    username, userid, usersec = user['name'], user['id'], user['section']

    # Delete face folder
    folder = f'static/faces/{username}${userid}${usersec}'
    if os.path.exists(folder):
        shutil.rmtree(folder)
        train_model()

    # Delete from DB
    cur.execute("DELETE FROM student WHERE id = ? and status='registered'", (userid,))
    conn.commit()
    conn.close()

    # Log action
    log_admin_action(session['admin'], 'DELETE USER', f'Deleted registered user {userid} ({username})')

    # Refresh list
    return redirect(url_for('register_user_list'))

# ======== Flask Login =========
@app.route('/login', methods=['GET', 'POST'])
def login():
    if g.user:
        session.pop('admin', None)
        return redirect(url_for('home', admin=False))

    if request.method == 'POST':
        admin_id = request.form['admin_id']
        password = request.form['password']

        conn = get_db()
        cur = conn.cursor()

        # Step 1: Fetch user
        cur.execute("SELECT * FROM admin_signup WHERE admin_id = ?", (admin_id,))
        user = cur.fetchone()

        if user and check_password_hash(user['password'], password):
            # Step 2: Insert login record (store login time, not password!)
            try:
                cur.execute("INSERT INTO admin_login (admin_id, username) VALUES (?, ?)",
                            (admin_id, user['username']))
                conn.commit()
                print("Stored login record for:", admin_id)
                
                # Log action
                try:
                    cur.execute("INSERT INTO admin_action_log (admin_id, action, details) VALUES (?, ?, ?)", 
                               (admin_id, 'LOGIN', 'Admin logged in'))
                    conn.commit()
                except Exception as ex:
                    print(f"Error logging login action: {ex}")
                    
            except Exception as e:
                conn.rollback()
                print("!!!Error inserting login record:", e)

            # Step 3: Save session
            session['admin'] = admin_id
            conn.close()
            return redirect(url_for('home', admin=True, mess=f'Logged in as {admin_id}'))
        else:
            conn.close()
            return render_template('LogInForm.html', mess='Incorrect Admin ID or Password')

    return render_template('LogInForm.html')

# ======== Flask Logout =========
@app.route('/logout')
def logout():
    session.pop('admin', None)
    return render_template('LogInForm.html')

# ======== Flask Sign Up =========
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    
    if request.method == 'POST':
        admin_id = request.form['admin_id']
        username = request.form['username']
        password = request.form['password']
        hashed_password = generate_password_hash(password)

        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT * FROM admin_signup WHERE admin_id = ?", (admin_id,))
        existing_user = cur.fetchone()

        # Check if user already exists
        if existing_user:
            conn.close()
            mess = "Admin ID already exists!"
            return render_template('SignUpPage.html', mess=mess)

        try:
            # Insert new user
            cur.execute("INSERT INTO admin_signup (admin_id, username, password) VALUES (?, ?, ?)",
                        (admin_id, username, hashed_password))
            
            # Log action
            cur.execute("INSERT INTO admin_action_log (admin_id, action, details) VALUES (?, ?, ?)", 
                       (admin_id, 'SIGNUP', 'New admin account created'))
            
            conn.commit()
            mess = "Account created successfully! Please log in."
            conn.close()
            return redirect(url_for('login', mess=mess))
        except Exception as e:
            conn.rollback()
            mess = f"Database error: {str(e)}"
            conn.close()
            return render_template('SignUpPage.html', mess=mess)

    return render_template('SignUpPage.html')

@app.route('/adminlog')
def adminlog():
    # Ensure admin is logged in
    if 'admin' not in session:
        return render_template('LogInForm.html', mess="Please log in first.")

    conn = get_db()
    cur = conn.cursor()
    
    # Fetch all admins for the management list
    cur.execute("SELECT * FROM admin_signup")
    all_admins = cur.fetchall()
    
    # Fetch all admin actions
    cur.execute("""
        SELECT l.admin_id, s.username, l.action, l.details, l.timestamp
        FROM admin_action_log l
        LEFT JOIN admin_signup s ON l.admin_id = s.admin_id
        ORDER BY l.id DESC
    """)
    logs = cur.fetchall()
    conn.close()

    admin_ids = [log['admin_id'] for log in logs]
    usernames = [log['username'] if log['username'] else 'Deleted Admin' for log in logs]
    actions = [log['action'] for log in logs]
    details = [log['details'] for log in logs]
    timestamps = [log['timestamp'] for log in logs]

    return render_template('AdminLog.html', 
                           admin_ids=admin_ids, 
                           usernames=usernames, 
                           actions=actions, 
                           details=details, 
                           timestamps=timestamps, 
                           l=len(logs),
                           all_admins=all_admins,
                           current_admin=session['admin'])

@app.route('/delete_admin', methods=['POST'])
def delete_admin():
    if 'admin' not in session:
        return redirect(url_for('login'))
        
    target_admin_id = request.form['admin_id']
    current_admin_id = session['admin']
    
    # Prevent self-deletion
    if target_admin_id == current_admin_id:
        return redirect(url_for('adminlog', mess="Cannot delete your own account!"))
        
    conn = get_db()
    cur = conn.cursor()
    
    try:
        # Delete the admin
        cur.execute("DELETE FROM admin_signup WHERE admin_id = ?", (target_admin_id,))
        
        # Log the action
        log_admin_action(current_admin_id, 'DELETE ADMIN', f'Deleted admin account: {target_admin_id}')
        
        conn.commit()
        mess = f"Admin {target_admin_id} deleted successfully."
    except Exception as e:
        conn.rollback()
        mess = f"Error deleting admin: {str(e)}"
    finally:
        conn.close()
        
    return redirect(url_for('adminlog', mess=mess))

# Main Function
if __name__ == '__main__':
    # Initialize database on startup
    init_db()
    
    # Get local IP address to show user
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        print(f"\n{'='*50}")
        print(f"Server is running!")
        print(f"To access from this computer: http://localhost:5001")
        print(f"To access from other devices: http://{local_ip}:5001")
        print(f"{'='*50}\n")
    except:
        print("Could not determine local IP. Try 'ipconfig' in terminal.")

    # Run on all available network interfaces (0.0.0.0) to allow access from other devices on the same LAN
    app.run(host='0.0.0.0', port=5001, debug=True)