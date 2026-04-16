# 🌍 GeoSentinel AI: Smart Landslide Detection System

<p align="center">
<img src="https://img.shields.io/badge/Python-3.9-blue">
<img src="https://img.shields.io/badge/Framework-YOLOv8-green">
<img src="https://img.shields.io/badge/UI-Streamlit-orange">
<img src="https://img.shields.io/badge/Task-Landslide%20Detection-red">
</p>

---

## 📑 Table of Contents

* [Project Overview](#-project-overview)
* [Objectives](#-objectives)
* [Dataset Description](#-dataset-description)
* [Methodology](#-methodology)
* [System Architecture](#-system-architecture)
* [Features](#-features)
* [Installation](#-installation)
* [Usage](#-usage)
* [Results](#-results)
* [Failure Cases](#-failure-cases)
* [Future Improvements](#-future-improvements)
* [Conclusion](#-conclusion)
* [References](#-references)
* [Author](#-author)

---

# 📌 Project Overview

**GeoSentinel AI** is an intelligent landslide detection and alert system that uses deep learning and computer vision to identify landslides from images and drone videos.

The system integrates:

* 🧠 YOLOv8 segmentation model  
* 🗺️ Real-time map visualization (Folium)  
* 📩 Alert system (SMS + Email)  
* 🎥 Video frame analysis  
* 🔊 Voice alerts  

---

# 🎯 Objectives

* Detect landslides using YOLOv8 segmentation  
* Build a real-time alert system  
* Integrate map visualization  
* Support image & video inputs  
* Improve disaster response  

---

# 📂 Dataset Description

* Satellite images  
* Drone footage  
* Terrain images  

### Annotation

YOLO segmentation format (polygon masks)

### Augmentation

* Flipping  
* Rotation  
* Brightness adjustment  
* Mosaic  

---

# ⚙️ Methodology

### 1. Dataset Preparation
* Annotated using Roboflow  
* Polygon masks used  

### 2. Model Training
* Model: YOLOv8 Segmentation  
* Metrics: mAP, IoU  

### 3. App Development
* Built using Streamlit  
* Dark UI + responsive design  

### 4. Alert System
* SMS → Twilio  
* Email → SendGrid  

### 5. Testing
* Image inputs  
* Video inputs  
* Real-world evaluation  

---

# 🧠 System Architecture
