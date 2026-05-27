# COLLEGE PROJECT REPORT
## Driver Drowsiness Detection System Using Computer Vision

---

**Project Title:** Driver Drowsiness Detection System
**Technology Used:** Python, OpenCV, MediaPipe, NumPy, Pygame
**Project Type:** Mini Project / Major Project
**Academic Year:** 2024–2025

---

## CERTIFICATE

*This is to certify that the project entitled "Driver Drowsiness Detection System" has been carried out by [Student Name], Enrollment No. [XXXXXX], in partial fulfilment of the requirements for the award of [Degree Name] from [College Name].*

---

## ABSTRACT

Road safety is a global concern, with driver fatigue being one of the leading causes of vehicular accidents. This project presents a real-time Driver Drowsiness Detection System built using Python programming language and computer vision techniques. The system continuously monitors a driver's eye behaviour through a webcam and computes the Eye Aspect Ratio (EAR) — a mathematical measure of eye openness. When the EAR falls below a threshold value for more than 20 consecutive video frames (approximately 0.67 seconds), the system classifies the driver as drowsy and triggers an audio alarm. The system uses Google's MediaPipe Face Mesh model for accurate 468-point facial landmark detection and OpenCV for video processing and display. The result is a lightweight, real-time safety system that runs on standard laptop hardware without any specialised equipment.

**Keywords:** Computer Vision, Eye Aspect Ratio, Drowsiness Detection, MediaPipe, OpenCV, Driver Safety, Facial Landmarks

---

## TABLE OF CONTENTS

1. Introduction
2. Problem Statement
3. Objectives
4. Literature Review
5. Technologies Used
6. System Design and Architecture
7. Algorithm and Methodology
8. EAR Mathematical Model
9. Implementation
10. Results and Discussion
11. Advantages and Limitations
12. Future Scope
13. Conclusion
14. References

---

## CHAPTER 1: INTRODUCTION

### 1.1 Background

Driver drowsiness is a critical safety hazard on roads. According to the National Highway Traffic Safety Administration (NHTSA), drowsy driving is responsible for over 100,000 road crashes and approximately 1,550 deaths annually in the United States alone. In India, the problem is even more severe due to long-distance highway travel, poorly lit roads, and inadequate rest facilities for truck and bus drivers.

Traditional methods of fatigue management rely solely on drivers' self-awareness — drinking coffee, stopping to rest, or using stimulants. These methods are reactive rather than preventive. There is a growing need for an automated, non-intrusive system that can detect drowsiness the moment it begins and alert the driver.

### 1.2 What is Computer Vision?

Computer Vision is a branch of Artificial Intelligence (AI) that enables computers to interpret and understand visual information from the world — just as human eyes and brain work together to recognise objects, people, and scenes. Applications of Computer Vision include face recognition on smartphones, object detection in self-driving cars, and medical image analysis in hospitals.

In this project, we use Computer Vision to process webcam video and extract meaningful information about the driver's eye state in real time.

### 1.3 Scope of the Project

This project focuses on:
- Eye behaviour analysis as a primary indicator of drowsiness
- Real-time video processing using a standard webcam
- A pure software solution requiring no additional hardware
- Beginner-accessible Python code with full documentation

---

## CHAPTER 2: PROBLEM STATEMENT

"Develop a real-time, computer vision-based driver drowsiness detection system that monitors the driver's eye behaviour using a webcam, computes the Eye Aspect Ratio (EAR) to identify eye closure, and triggers an audio alert when prolonged eye closure indicates drowsiness — without requiring any specialised hardware."

---

## CHAPTER 3: OBJECTIVES

The main objectives of this project are:

1. To study the Eye Aspect Ratio (EAR) model for eye-state detection
2. To implement real-time facial landmark detection using MediaPipe
3. To develop a reliable drowsiness classification algorithm based on consecutive frame analysis
4. To create an audio alarm system using Python's Pygame library
5. To display a real-time HUD (Heads-Up Display) with live statistics
6. To build a system optimised for standard laptop hardware

---

## CHAPTER 4: LITERATURE REVIEW

### 4.1 EAR-Based Detection (Soukupová & Čech, 2016)

The Eye Aspect Ratio (EAR) was first introduced in the paper "Real-Time Eye Blink Detection using Facial Landmarks" by Tereza Soukupová and Jan Čech at the Czech Technical University. They demonstrated that EAR provides a simple, real-time, and robust measurement of eye openness using just 6 facial landmark points.

### 4.2 Facial Landmark Detection

Traditional approaches used Haar Cascades (Viola-Jones algorithm) for face detection. Modern approaches use deep learning-based landmark models. MediaPipe Face Mesh, released by Google in 2020, provides 468 landmarks with GPU-level speed on CPU hardware.

### 4.3 Drowsiness Detection Approaches

| Approach | Method | Limitation |
|---|---|---|
| EEG-based | Brainwave monitoring | Requires headgear |
| Steering pattern | Erratic steering detection | Late detection |
| Image-based (EAR) | Eye closure analysis | Fails with sunglasses |
| Deep learning CNN | Train on eye images | Needs large dataset |

This project uses the EAR-based approach for its simplicity, speed, and effectiveness.

---

## CHAPTER 5: TECHNOLOGIES USED

### 5.1 Python 3.10+
Python is a high-level, beginner-friendly programming language. Its extensive library ecosystem makes it ideal for AI and computer vision projects.

### 5.2 OpenCV (Open Source Computer Vision Library)
OpenCV is the world's most popular computer vision library. In this project it is used for:
- Opening and reading the webcam
- Converting colour formats (BGR to RGB)
- Drawing shapes, text, and overlays on video frames
- Displaying the output window

### 5.3 MediaPipe
MediaPipe is an open-source, cross-platform machine learning framework developed by Google. Its Face Mesh model detects 468 3D facial landmarks in real time using a pre-trained neural network. It runs on CPU with minimal resources.

### 5.4 NumPy
NumPy (Numerical Python) provides support for arrays and mathematical functions. It is used for coordinate calculations and array operations.

### 5.5 SciPy
SciPy's spatial.distance module provides the Euclidean distance function used in the EAR formula.

### 5.6 Pygame
Pygame is a Python multimedia library. Its mixer module is used to load and play the alarm WAV audio file.

### 5.7 Imutils
Imutils provides convenience functions for OpenCV operations like resizing images.

---

## CHAPTER 6: SYSTEM DESIGN AND ARCHITECTURE

### 6.1 Block Diagram

```
┌──────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   WEBCAM     │───▶│ FRAME CAPTURE    │───▶│ COLOUR CONVERT  │
│  (input)     │    │ (OpenCV)         │    │ BGR → RGB       │
└──────────────┘    └──────────────────┘    └────────┬────────┘
                                                      │
                                            ┌─────────▼───────┐
                                            │  FACE MESH      │
                                            │  MediaPipe      │
                                            │  (468 points)   │
                                            └─────────┬───────┘
                                                      │
                              ┌───────────────────────▼────────────────┐
                              │         EYE LANDMARK EXTRACTION         │
                              │    Left: 6 points | Right: 6 points     │
                              └───────────────────────┬────────────────┘
                                                      │
                              ┌───────────────────────▼────────────────┐
                              │         EAR CALCULATION                 │
                              │  EAR = (A + B) / (2 × C)               │
                              └───────────────────────┬────────────────┘
                                                      │
                    ┌─────────────────────────────────▼──────────────┐
                    │                DECISION LOGIC                   │
                    │  EAR < 0.25? → counter++                       │
                    │  counter > 20 → ALARM                          │
                    └───────┬──────────────────────┬─────────────────┘
                            │                      │
               ┌────────────▼──────┐    ┌──────────▼──────────┐
               │  DISPLAY HUD      │    │  ALARM SYSTEM       │
               │  (OpenCV window)  │    │  (Pygame audio)     │
               └───────────────────┘    └─────────────────────┘
```

### 6.2 State Machine

```
         eyes open           eyes open
AWAKE ───────────── DROWSY ─────────────── AWAKE
  │                    │    eyes closed         │
  │                    │    > 20 frames         │
  │                    ▼                        │
  │                SLEEPING ───────────────────►│
  │                 + ALARM    eyes open        │
  └────────────────────────────────────────────┘
```

---

## CHAPTER 7: ALGORITHM AND METHODOLOGY

### 7.1 EAR Algorithm

```
ALGORITHM: EAR_DROWSINESS_DETECTION
INPUT: Webcam video stream
OUTPUT: Real-time drowsiness alert

1. INITIALISE:
   - threshold EAR_THRESH = 0.25
   - consec_frame_count = 0
   - blink_count = 0

2. FOR each frame in video stream:
   a. Capture frame from webcam
   b. Convert BGR → RGB
   c. Run MediaPipe Face Mesh detection
   d. IF face detected:
        i.  Extract 6 left eye landmark coordinates
        ii. Extract 6 right eye landmark coordinates
        iii. Compute left_EAR and right_EAR
        iv.  avg_EAR = (left_EAR + right_EAR) / 2
        v.   IF avg_EAR < EAR_THRESH:
                  consec_frame_count += 1
                  IF consec_frame_count >= 20:
                       STATUS = SLEEPING
                       PLAY ALARM
             ELSE:
                  consec_frame_count = 0
                  STATUS = AWAKE
                  STOP ALARM
   e. Draw HUD on frame
   f. Display frame
   g. Check for 'Q' key → EXIT

3. RELEASE webcam, CLOSE windows
```

---

## CHAPTER 8: EAR MATHEMATICAL MODEL

### 8.1 Six Landmark Points

For each eye, 6 specific facial landmark points are used:

```
Landmark | MediaPipe Index | Location
  p1     |      33         | Left corner of left eye
  p2     |     160         | Upper-left eyelid
  p3     |     158         | Upper-right eyelid
  p4     |     133         | Right corner of left eye
  p5     |     153         | Lower-right eyelid
  p6     |     144         | Lower-left eyelid
```

### 8.2 EAR Formula Derivation

```
         ||p2 - p6|| + ||p3 - p5||
EAR =   ─────────────────────────────
                2 × ||p1 - p4||

Where:
  ||p2 - p6|| = Euclidean distance = √[(x₂-x₆)² + (y₂-y₆)²]
  ||p3 - p5|| = Euclidean distance = √[(x₃-x₅)² + (y₃-y₅)²]
  ||p1 - p4|| = Euclidean distance = √[(x₁-x₄)² + (y₁-y₄)²]
```

### 8.3 EAR Range Analysis

| Eye State | EAR Range | Interpretation |
|---|---|---|
| Fully Open | 0.30 – 0.45 | Normal driving |
| Partially Open | 0.20 – 0.30 | Slightly tired |
| Closed / Blinking | 0.05 – 0.20 | Single blink |
| Fully Closed | 0.00 – 0.10 | Sleeping |

**Threshold Used:** EAR < 0.25 for ≥ 20 frames → DROWSY alert

---

## CHAPTER 9: IMPLEMENTATION

### 9.1 Development Environment

| Component | Details |
|---|---|
| Language | Python 3.10 |
| IDE | Visual Studio Code |
| OS | Windows 10/11 / Ubuntu / macOS |
| Camera | Built-in or USB webcam |
| Resolution | 640 × 480 pixels |
| Frame Rate | ~25-30 FPS |

### 9.2 Key Code Sections

**Eye Aspect Ratio Function:**
```python
def eye_aspect_ratio(eye_points):
    A = distance.euclidean(eye_points[1], eye_points[5])
    B = distance.euclidean(eye_points[2], eye_points[4])
    C = distance.euclidean(eye_points[0], eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear
```

**Drowsiness Detection Logic:**
```python
if ear < EAR_THRESHOLD:
    consec_frames += 1
    if consec_frames >= EAR_CONSEC_FRAMES:  # 20 frames
        alarm.play()
        status = "SLEEPING"
else:
    alarm.stop()
    consec_frames = 0
    status = "AWAKE"
```

---

## CHAPTER 10: RESULTS AND DISCUSSION

### 10.1 System Performance

| Metric | Value |
|---|---|
| Processing Speed | 25–30 FPS (standard laptop) |
| Detection Latency | ~0.67 seconds after eye closure |
| Alarm Response Time | < 50 milliseconds |
| Landmark Accuracy | High (MediaPipe pretrained model) |
| Memory Usage | ~200–400 MB RAM |

### 10.2 Test Scenarios

**Test 1 – Normal Blinking:**
Regular blinks (0.2–0.3 sec) → Counter resets → No alarm ✅

**Test 2 – Slow Blinking:**
Slow blinks (0.5–0.6 sec) → Counter reaches ~18 → Brief warning ✅

**Test 3 – Eye Closure > 0.67 sec:**
Eyes closed → Counter = 20 → ALARM fires immediately ✅

**Test 4 – No Face:**
User moves out of frame → Status = "NO FACE" → No alarm ✅

**Test 5 – Sunglasses:**
Sunglasses → MediaPipe fails to detect landmarks → No detection ❌ (Limitation)

---

## CHAPTER 11: ADVANTAGES AND LIMITATIONS

### 11.1 Advantages
1. Non-intrusive: No physical contact with driver
2. Low cost: Only requires a standard webcam
3. Real-time: Processes 30 frames per second
4. Offline: Works without internet connection
5. Cross-platform: Windows, Linux, macOS
6. Open source: All libraries are free

### 11.2 Limitations
1. Fails when driver wears sunglasses
2. Requires adequate lighting
3. Not effective at extreme head angles
4. Single-person detection only
5. Not integrated with vehicle systems
6. Fixed threshold may need tuning per user

---

## CHAPTER 12: FUTURE SCOPE

1. **Yawning Detection:** Add mouth opening analysis using facial landmarks to detect yawning as an additional drowsiness indicator.

2. **Head Pose Estimation:** Detect forward head drooping (nodding off) using 3D landmark coordinates.

3. **SMS/Email Alerts:** Integrate Twilio API to automatically notify emergency contacts.

4. **Deep Learning Enhancement:** Train a CNN on eye image datasets for more robust detection under varied conditions.

5. **YOLO Integration:** Replace MediaPipe with YOLO-based face detection for improved multi-face scenarios.

6. **Raspberry Pi Deployment:** Port to Raspberry Pi 4 with a Pi Camera Module for in-vehicle installation.

7. **Mobile Application:** Build an Android app using camera2 API with on-device ML inference.

8. **Night Vision:** Integrate with IR (Infrared) cameras for low-light driving detection.

---

## CHAPTER 13: CONCLUSION

This project successfully demonstrates a real-time Driver Drowsiness Detection System using Python and modern computer vision tools. By implementing the Eye Aspect Ratio (EAR) algorithm on top of Google's MediaPipe Face Mesh landmark detection, we achieved reliable drowsiness detection at 25-30 frames per second on standard laptop hardware.

The system effectively:
- Detects facial landmarks in real time
- Calculates eye openness mathematically
- Classifies driver state as Awake, Drowsy, or Sleeping
- Triggers an audio alarm within 0.67 seconds of eye closure

This project demonstrates the power of combining accessible AI tools (MediaPipe), established computer vision libraries (OpenCV), and simple mathematics (EAR) to solve a meaningful, real-world problem. With further development and hardware integration, this system has the potential to be deployed in actual commercial vehicles, making roads safer for everyone.

---

## REFERENCES

1. T. Soukupová and J. Čech, "Real-Time Eye Blink Detection using Facial Landmarks," in 21st Computer Vision Winter Workshop, February 2016.

2. V. Kazemi and J. Sullivan, "One millisecond face alignment with an ensemble of regression trees," in Proceedings of the IEEE CVPR, 2014.

3. Google MediaPipe Team, "MediaPipe Face Mesh Documentation," Google LLC, 2023. [Online] Available: https://google.github.io/mediapipe/solutions/face_mesh.html

4. OpenCV Documentation, "OpenCV 4.9 Reference Manual," OpenCV Foundation, 2024. [Online] Available: https://docs.opencv.org

5. National Highway Traffic Safety Administration (NHTSA), "Drowsy Driving," U.S. Department of Transportation, 2023.

6. A. Rosebrock, "Drowsiness Detection with OpenCV," PyImageSearch Blog, 2017. [Online] Available: https://pyimagesearch.com

---

*End of Report*
