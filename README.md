# 🎓 Face Recognition Attendance Machine Engine
<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.0+-000000?style=for-the-badge&logo=flask&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-3.0+-003B57?style=for-the-badge&logo=sqlite&logoColor=white)

**An intelligent attendance tracking system powered by deep learning and computer vision**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Tech Stack](#-tech-stack) • [Architecture](#-architecture)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [API Endpoints](#-api-endpoints)
- [Database Schema](#-database-schema)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

The **Face Recognition Attendance System** is an automated attendance tracking solution that leverages state-of-the-art deep learning models to identify and record student attendance in real-time. The system uses a Convolutional Neural Network (CNN) trained on facial features to recognize registered students through a webcam interface.

### Key Highlights

- 🤖 **AI-Powered Recognition**: CNN-based face recognition with high accuracy
- 📸 **Real-Time Detection**: Live webcam feed with instant face detection
- 🔒 **Secure Authentication**: Admin login system with password hashing
- 📊 **Comprehensive Reports**: View attendance by date, student ID, or section
- 👥 **User Management**: Register, unregister, and manage student profiles
- 🎨 **Modern Web Interface**: Clean and intuitive Flask-based UI

---

## ✨ Features

### Core Functionality

- **👤 Face Registration**
  - Capture up to 100 face images per student
  - Automatic face and eye detection validation
  - Real-time feedback during image capture

- **🎯 Attendance Marking**
  - Real-time face recognition via webcam
  - Temporal smoothing to prevent false positives
  - Automatic duplicate prevention (one attendance per day)
  - Visual feedback with bounding boxes and labels

- **📈 Attendance Management**
  - View today's attendance records
  - Search attendance by specific date
  - Filter attendance by student ID
  - Export attendance data

- **👨‍💼 Admin Dashboard**
  - Secure admin authentication
  - Student registration and unregistration
  - Manage registered and unregistered students
  - View admin login logs

- **🧠 Intelligent Model Training**
  - Automatic CNN model training on new registrations
  - Data augmentation for improved accuracy
  - Model validation and retraining on class mismatches

---

## 🛠️ Tech Stack

### Backend
<div align="left">

| Technology | Version | Purpose |
|------------|---------|---------|
| ![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white) | 3.8+ | Core programming language |
| ![Flask](https://img.shields.io/badge/Flask-2.0+-000000?style=flat-square&logo=flask&logoColor=white) | 2.0+ | Web framework |
| ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-FF6F00?style=flat-square&logo=tensorflow&logoColor=white) | 2.0+ | Deep learning framework |
| ![Keras](https://img.shields.io/badge/Keras-D00000?style=flat-square&logo=keras&logoColor=white) | Built-in | High-level neural network API |
| ![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-5C3EE8?style=flat-square&logo=opencv&logoColor=white) | 4.0+ | Computer vision library |
| ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white) | Latest | Numerical computing |
| ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white) | Latest | Data manipulation |
| ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white) | 1.7.1 | Machine learning utilities |

</div>

### Database
<div align="left">

| Technology | Purpose |
|------------|---------|
| ![SQLite](https://img.shields.io/badge/SQLite-3.0+-003B57?style=flat-square&logo=sqlite&logoColor=white) | Lightweight relational database |

</div>

### Security & Utilities
<div align="left">

| Technology | Purpose |
|------------|---------|
| ![Werkzeug](https://img.shields.io/badge/Werkzeug-000000?style=flat-square&logo=werkzeug&logoColor=white) | Password hashing and security |
| ![python-dotenv](https://img.shields.io/badge/python--dotenv-000000?style=flat-square) | Environment variable management |
| ![Pillow](https://img.shields.io/badge/Pillow-8.0+-013243?style=flat-square&logo=pillow&logoColor=white) | Image processing |

</div>

### Frontend
<div align="left">

| Technology | Purpose |
|------------|---------|
| ![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=flat-square&logo=html5&logoColor=white) | Markup language |
| ![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=flat-square&logo=css3&logoColor=white) | Styling |
| ![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat-square&logo=javascript&logoColor=black) | Client-side scripting |
| ![Jinja2](https://img.shields.io/badge/Jinja2-B41717?style=flat-square&logo=jinja&logoColor=white) | Template engine |

</div>

---

## 🏗️ Architecture

### System Flow

```
┌─────────────────┐
│   Web Browser   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Flask App     │ ◄─── Session Management
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌─────────┐ ┌──────────────┐
│ OpenCV  │ │  TensorFlow  │
│ Camera  │ │  CNN Model   │
└────┬────┘ └──────┬───────┘
     │             │
     └──────┬──────┘
            │
            ▼
     ┌──────────┐
     │ SQLite   │
     │ Database │
     └──────────┘
```

### Face Recognition Pipeline

1. **Face Detection**: Haar Cascade classifiers detect faces and eyes in video frames
2. **Preprocessing**: Resize and normalize face images to 224×224 pixels
3. **Feature Extraction**: CNN model extracts facial features
4. **Classification**: Softmax layer predicts student identity with confidence scoring
5. **Temporal Smoothing**: Consecutive frame analysis prevents false positives
6. **Attendance Recording**: Validated identities are logged to the database

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- Webcam/camera device
- pip package manager

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd FRAME
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirement.txt
```

### Step 4: Environment Setup

Create a `.env` file in the project root:

```env
SECRET_KEY=your-secret-key-here
```

### Step 5: Initialize Database

The database will be automatically created on first run. Ensure the following directories exist:

```
FRAME/
├── static/
│   ├── faces/          # Student face images
│   └── haarcascade_*.xml  # Haar cascade files
└── final_model/        # Trained CNN models
```

### Step 6: Run the Application

```bash
python app.py
```

The application will be available at `http://localhost:5001`

---

## 🚀 Usage

### 1. Admin Registration

- Navigate to the Sign Up page
- Create an admin account with Admin ID, username, and password
- Login credentials are securely hashed using Werkzeug

### 2. Student Registration

1. **Add New User**:
   - Click "Add New User" from the homepage
   - Enter student name, ID, and section
   - Position face in front of webcam
   - System captures 100 face images automatically
   - Press `ESC` to stop early

2. **Register Student**:
   - Admin logs in
   - View unregistered students list
   - Assign section and register students
   - Model automatically retrains with new data

### 3. Mark Attendance

1. Navigate to "Take Attendance"
2. Click "Take Attendance" button
3. Position face in front of webcam
4. System recognizes and marks attendance automatically
5. Press `ESC` to close camera window
6. View attendance records on the attendance page

### 4. View Attendance Reports

- **Today's Attendance**: Automatically displayed on attendance page
- **By Date**: Admin can search attendance for specific dates
- **By Student ID**: Filter attendance records by student ID
- **Export**: Attendance data can be exported from the database

### 5. Manage Students

- **Unregister**: Move registered students to unregistered list
- **Delete**: Permanently remove students from the system
- **View Lists**: Separate views for registered and unregistered students

---

## 📁 Project Structure

```
FRAME/
│
├── app.py                          # Main Flask application
├── requirement.txt                 # Python dependencies
├── .env                           # Environment variables (create this)
├── .gitignore                     # Git ignore rules
│
├── static/                        # Static files
│   ├── faces/                     # Student face image directories
│   │   └── [Name]$[ID]$[Section]/ # Individual student folders
│   ├── haarcascade_frontalface_default.xml  # Face detection model
│   ├── haarcascade_eye.xml        # Eye detection model
│   └── images/                    # UI images and icons
│
├── templates/                     # HTML templates
│   ├── HomePage.html              # Landing page
│   ├── AddUser.html               # User registration page
│   ├── Attendance.html            # Attendance marking interface
│   ├── AttendanceList.html        # Attendance records view
│   ├── RegisterUserList.html      # Registered students list
│   ├── UnregisterUserList.html    # Unregistered students list
│   ├── LogInForm.html             # Admin login page
│   ├── SignUpPage.html            # Admin signup page
│   ├── AdminLog.html              # Admin login logs
│   └── Error.html                 # Error page
│
├── final_model/                   # Trained models
│   ├── face_recognition_model.h5  # CNN model weights
│   └── class_names.pkl            # Class name mappings
│
└── attendance.db                  # SQLite database (auto-generated)
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| `GET` | `/` | Home page | No |
| `GET` | `/attendance` | View attendance page | No |
| `GET` | `/attendancebtn` | Start attendance marking | No |
| `GET` | `/adduser` | Add new user page | No |
| `POST` | `/adduserbtn` | Process user registration | No |
| `GET` | `/attendancelist` | View today's attendance | Yes |
| `POST` | `/attendancelistdate` | Search attendance by date | Yes |
| `POST` | `/attendancelistid` | Search attendance by ID | Yes |
| `GET` | `/registeruserlist` | View registered students | Yes |
| `POST` | `/registeruser` | Register a student | Yes |
| `POST` | `/unregisteruser` | Unregister a student | Yes |
| `GET` | `/unregisteruserlist` | View unregistered students | Yes |
| `POST` | `/deleteregistereduser` | Delete registered student | Yes |
| `POST` | `/deleteunregistereduser` | Delete unregistered student | Yes |
| `GET` | `/login` | Login page | No |
| `POST` | `/login` | Process login | No |
| `GET` | `/logout` | Logout user | No |
| `GET` | `/signup` | Signup page | No |
| `POST` | `/signup` | Process signup | No |
| `GET` | `/adminlog` | View admin login logs | Yes |

---

## 🗄️ Database Schema

### `student` Table
| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT (PK) | Student ID |
| `name` | TEXT | Student name |
| `section` | TEXT | Section/Class |
| `status` | TEXT | 'registered' or 'unregistered' |

### `attendance` Table
| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT (FK) | Student ID |
| `name` | TEXT | Student name |
| `section` | TEXT | Section/Class |
| `time` | TEXT | Timestamp (YYYY-MM-DD HH:MM:SS) |

### `admin_signup` Table
| Column | Type | Description |
|--------|------|-------------|
| `admin_id` | TEXT (PK) | Admin ID |
| `username` | TEXT | Admin username |
| `password` | TEXT | Hashed password |

### `admin_login` Table
| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER (PK) | Auto-increment ID |
| `admin_id` | TEXT (FK) | Admin ID |
| `username` | TEXT | Admin username |
| `login_time` | TEXT | Login timestamp |

---

## 🎨 Model Architecture

The CNN model used for face recognition:

```
Input Layer: (224, 224, 3)
    ↓
Conv2D (32 filters, 3×3) + ReLU
    ↓
MaxPooling2D (2×2)
    ↓
Conv2D (64 filters, 3×3) + ReLU
    ↓
MaxPooling2D (2×2)
    ↓
Conv2D (128 filters, 3×3) + ReLU
    ↓
MaxPooling2D (2×2)
    ↓
Flatten
    ↓
Dense (256 units) + ReLU
    ↓
Dropout (0.5)
    ↓
Dense (num_classes) + Softmax
```

**Training Parameters:**
- Optimizer: Adam (learning rate: 0.001)
- Loss: Categorical Crossentropy
- Epochs: 20
- Batch Size: 32
- Data Augmentation: Rotation, shifts, shear, zoom, flip, brightness

---

## 🔧 Configuration

### Model Settings

- **Confidence Threshold**: 0.7 (minimum confidence for recognition)
- **Margin Threshold**: 0.15 (separation from runner-up)
- **Consecutive Frames**: 5 (required for attendance marking)
- **Image Size**: 224×224 pixels
- **Max Images per User**: 100

### Camera Settings

- **Camera Index**: 0 (default webcam)
- **Face Detection**: Haar Cascade (frontal face)
- **Eye Detection**: Haar Cascade (eyes)
- **Minimum Face Size**: 100×100 pixels

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add comments for complex logic
- Update documentation for new features
- Test thoroughly before submitting PR

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

## 👥 Authors

- **Raka Maharjan** - *Initial work* - [YourGitHub](https://github.com/blackST4Rez)
- **Shuvam Shakya** - *Initial work* - [YourGitHub](https://github.com/Shuvam02)

---

## 🙏 Acknowledgments

- OpenCV community for face detection algorithms
- TensorFlow team for deep learning framework
- Flask community for web framework
- All contributors and users of this project

---

<div align="center">

**Made with ❤️ using Python, Flask, and TensorFlow**

⭐ Star this repo if you find it helpful!

</div>
