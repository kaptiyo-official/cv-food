# 🌾 Crop Health Analyzer

A web-based tool to **analyze crop images** using OpenCV and **predict their health status**, built with **Python**, **Flask**, **HTML**, and **CSS**.

---

## 📌 Overview

This project allows users to upload an image of a crop, and the system:
- 🩺 Analyzes its health using image processing techniques
- 💡 Provides helpful tips to improve plant care or recovery

Built as part of a **2025 Python Bootcamp project**, with teamwork and collaboration.

---

## 🤝 Collaboration

This was a **2-person project** developed during a coding bootcamp in 2025.

- 💻 Both members contributed to **backend, frontend, and logic**
- 🔄 We helped each other debug, build, and refine the app
- 🤝 A shared learning experience in web + OpenCV development

---

## 🎥 Demo Video

[DEMO](https://youtu.be/r1Q4_GqOsCc)

---

## 🧠 Features

- 🖼 Upload interface to select crop images
- 🔍 Real-time OpenCV analysis of crop visuals
- 🌱 Health status output (Healthy / Diseased)
- 💡 Tips based on diagnosis
- 🧪 Modular structure for future improvements

---

## 🛠️ Tech Stack

| Layer     | Tools Used      |
|-----------|-----------------|
| Backend   | Python, Flask   |
| Frontend  | HTML, CSS       |
| Imaging   | OpenCV          |
| Templates | Jinja2 (Flask)  |

---

## 🚀 Getting Started

### 📦 Installation

Install required libraries:

```
pip install torch torchvision numpy opencv-python
```

### ▶️ Running the App

```
python app.py
```

Then open your browser and go to:

```
http://127.0.0.1:5000/
```

---

## 📂 Project Structure

```
crop-analyzer/
│
├── static/                  # CSS, images, etc.
├── templates/               # HTML templates
│   └── index.html
├── app.py                   # Main Flask app
├── crop_detection.py        # OpenCV logic
├── README.md
└── requirements.txt         # Dependencies
```

---

## 📌 Future Improvements

- 🤖 Replace basic logic with trained ML model
- 📱 Mobile-friendly responsive layout
- 🌍 Add multilingual crop info (Telugu, Hindi, etc.)

---

## 📜 License

MIT License — feel free to use, improve, and share.

---

## ❤️ Acknowledgements

Built during a 2025 bootcamp with collaboration and shared curiosity.

> *“Let’s help farmers one pixel at a time.”*