# 🚗 Driver Drowsiness Detection System

> **Real-time AI-based drowsiness detection using Python, OpenCV & MediaPipe**
> A beginner-friendly college mini project with complete documentation.

---

## 📁 Project Structure

```
drowsiness_detection/
│
├── main.py               ← Main program (run this!)
├── generate_alarm.py     ← Run once to create alarm.wav
├── alarm.wav             ← Generated alarm sound file
├── requirements.txt      ← Python library dependencies
└── README.md             ← This file
```

---

## ⚡ Quick Start (Run in 4 steps)

```bash
# Step 1 – Go to project folder
cd drowsiness_detection

# Step 2 – Install all libraries
pip install -r requirements.txt

# Step 3 – Generate the alarm sound
python generate_alarm.py

# Step 4 – Launch the system!
python main.py
```

Press **Q** to quit | Press **R** to reset blink counter

---

## 📖 PROJECT DOCUMENTATION

---

### 1. PROJECT INTRODUCTION

Road accidents caused by drowsy drivers are one of the leading causes of death on highways worldwide. Studies show that driving while sleepy is as dangerous as drunk driving. This project presents a real-time system that monitors a driver's eyes using an ordinary laptop webcam and raises an alarm the moment the driver shows signs of falling asleep.

The system uses Computer Vision — the science of making computers "see" and understand images — combined with AI-powered facial landmark detection to track the driver's eyes 30 times every second. It requires no special hardware, just a laptop or PC with a webcam.

---

### 2. OBJECTIVES

- ✅ Detect driver drowsiness in real-time using a webcam
- ✅ Monitor eye opening/closing using facial landmarks
- ✅ Compute Eye Aspect Ratio (EAR) mathematically
- ✅ Sound an alarm when the driver's eyes remain closed too long
- ✅ Display live stats: FPS, EAR value, blink count, status
- ✅ Build a clean, beginner-friendly Python project

---

### 3. PROBLEM STATEMENT

"Driver fatigue and drowsiness are responsible for approximately 20% of all road accidents. There is a need for an automated, low-cost, real-time system that can detect driver drowsiness and alert the driver before an accident occurs."

---

### 4. METHODOLOGY

```
Webcam Feed
    ↓
Convert to RGB
    ↓
MediaPipe Face Mesh (468 landmark points)
    ↓
Extract 6 Eye Landmarks (Left + Right)
    ↓
Compute EAR (Eye Aspect Ratio)
    ↓
Is EAR < 0.25?
   YES → Increment counter
       → Counter > 20 frames? → ALARM + WARNING
   NO  → Reset counter, Status = AWAKE
    ↓
Draw HUD Dashboard on frame
    ↓
Display to screen
```

---

### 5. SYSTEM ARCHITECTURE

```
┌─────────────────────────────────────────────────────────┐
│                  INPUT LAYER                            │
│    Webcam (640×480 px, ~30 FPS)                         │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│               PROCESSING LAYER                          │
│  1. OpenCV – Frame capture & preprocessing              │
│  2. MediaPipe – Face Mesh (468 points, GPU/CPU)         │
│  3. NumPy / SciPy – EAR calculation                     │
│  4. State machine – Awake / Drowsy / Sleeping           │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                OUTPUT LAYER                             │
│  ● OpenCV window with HUD dashboard                     │
│  ● Pygame alarm (alarm.wav)                             │
│  ● Warning banner on screen                             │
└─────────────────────────────────────────────────────────┘
```

---

### 6. KEY CONCEPTS EXPLAINED (For Non-CS Students)

#### What is Computer Vision?
Computer Vision (CV) is a field of Artificial Intelligence that trains computers to understand images and videos — just like human eyes + brain. For example, when you upload a photo to Google Photos and it recognises your friend's face, that's Computer Vision.

#### What is OpenCV?
OpenCV (Open Source Computer Vision Library) is a free, open-source library (collection of ready-made tools) for image and video processing. It can:
- Open your webcam
- Read/write images and videos
- Draw shapes, text on images
- Detect faces, edges, colours

Think of OpenCV as a "Swiss army knife" for working with images in Python.

#### What is MediaPipe?
MediaPipe is a framework built by Google that uses AI/Deep Learning models to detect body parts in images. Its "Face Mesh" model can detect 468 specific points (called landmarks) on a human face — including the exact corners and edges of the eyes — in real time, even on a laptop CPU.

#### What is a Facial Landmark?
Imagine placing numbered dots on a face:
- Dot #33 = left corner of left eye
- Dot #160 = top of left eye
- Dot #133 = right corner of left eye
... and so on for 468 points total.

These dots are called "landmarks." By tracking where these dots are each frame, we can figure out if the eye is open or closed.

#### What is Eye Aspect Ratio (EAR)?
EAR is a simple number that tells us how OPEN an eye is:
- EAR ≈ 0.30 → Eye is fully OPEN
- EAR ≈ 0.10 → Eye is CLOSING
- EAR ≈ 0.00 → Eye is fully CLOSED

---

### 7. EAR MATHEMATICAL FORMULA

#### The 6 Eye Landmark Points

```
         p2 ─────── p3
        /               \
      p1                  p4
        \               /
         p6 ─────── p5
```

| Point | Location          |
|-------|------------------|
| p1    | Left corner       |
| p4    | Right corner      |
| p2    | Upper-left lid    |
| p3    | Upper-right lid   |
| p5    | Lower-right lid   |
| p6    | Lower-left lid    |

#### Formula

```
        ||p2 - p6|| + ||p3 - p5||
EAR =  ────────────────────────────
              2 × ||p1 - p4||
```

Where `||A - B||` means the Euclidean distance between two points:
```
||A - B|| = √[(x₂-x₁)² + (y₂-y₁)²]
```

#### Worked Example

Suppose (in pixels):
- p1 = (100, 200),  p4 = (160, 200)  → ||p1-p4|| = √(60²+0²) = **60**
- p2 = (110, 190),  p6 = (112, 212)  → ||p2-p6|| = √(4+484) ≈ **22.1**
- p3 = (148, 190),  p5 = (148, 212)  → ||p3-p5|| = √(0+484) ≈ **22.0**

```
EAR = (22.1 + 22.0) / (2 × 60) = 44.1 / 120 ≈ 0.368  → Eye is OPEN ✅
```

If the eye closes, p2,p3 move down toward p6,p5 → numerator → 0 → EAR → 0 ❌

#### Why is EAR Useful?
- It doesn't depend on the DISTANCE from the camera (scale-invariant)
- It doesn't depend on the HEAD ANGLE (works sideways too)
- It is VERY FAST to compute (just 3 distances + division)

---

### 8. ALGORITHMS USED

| Algorithm              | Purpose                              |
|------------------------|--------------------------------------|
| MediaPipe Face Mesh    | Detect 468 facial landmarks          |
| Euclidean Distance     | Measure distances between eye points |
| EAR Calculation        | Determine eye open/closed state      |
| Sliding Window Counter | Count consecutive closed frames      |
| State Machine          | AWAKE → DROWSY → SLEEPING transitions|

---

### 9. HOW THE ALARM WORKS

1. Every frame, EAR is computed.
2. If EAR < 0.25 (threshold), a frame counter goes UP by 1.
3. If this counter reaches 20 frames (~0.67 seconds):
   - Status changes to "SLEEPING"
   - Pygame plays alarm.wav on a loop
   - A red warning banner appears on screen
4. As soon as the driver opens their eyes:
   - EAR goes back above 0.25
   - Counter resets to 0
   - Alarm stops immediately

---

### 10. HOW TO RUN IN VS CODE

1. Open VS Code → File → Open Folder → select `drowsiness_detection/`
2. Open Terminal: View → Terminal (or Ctrl + `)
3. Run the following commands one by one:

```bash
pip install -r requirements.txt
python generate_alarm.py
python main.py
```

4. A window named "Driver Drowsiness Detection System" will open
5. Sit in front of your webcam and try closing your eyes for 1+ second

---

### 11. FIXING COMMON ERRORS

| Error Message | Solution |
|---|---|
| `ModuleNotFoundError: No module named 'cv2'` | Run `pip install opencv-python` |
| `ModuleNotFoundError: No module named 'mediapipe'` | Run `pip install mediapipe` |
| `Cannot access webcam` | Check webcam is connected; change `cv2.VideoCapture(0)` to `(1)` |
| `alarm.wav not found` | Run `python generate_alarm.py` first |
| `ImportError: DLL load failed` | Install Visual C++ Redistributable from Microsoft's website |
| `mediapipe not found on Python 3.12` | Use Python 3.10 or 3.11 instead |

---

### 12. EXPECTED OUTPUT

When running normally:
```
============================================================
  DRIVER DROWSINESS DETECTION SYSTEM
  Press [Q] to quit
============================================================
[INFO] Alarm sound loaded successfully.
[INFO] Starting webcam...
```

On screen you will see:
- 📷 Your webcam feed (mirrored like a selfie)
- 🟢 Green lines around your eyes when open
- 🔴 Red lines around your eyes when closed
- 📊 Right panel: FPS, EAR value, Blink Count, Status
- 📉 Left bar: EAR level indicator with threshold line
- 🚨 Red banner + beep alarm when eyes stay closed

---

### 13. ADVANTAGES

- ✅ Works on any standard laptop with webcam
- ✅ No special hardware required
- ✅ Real-time detection at 25-30 FPS
- ✅ Works in normal indoor lighting
- ✅ Free and open-source (zero cost)
- ✅ Can run offline (no internet needed)
- ✅ Beginner-friendly Python code

---

### 14. LIMITATIONS

- ❌ Performance drops in very low light
- ❌ Sunglasses will block eye detection
- ❌ Accuracy decreases with extreme head tilt
- ❌ Single camera angle (front-facing only)
- ❌ No GPS / vehicle speed integration
- ❌ Not tested for all face types / ethnicities equally

---

### 15. FUTURE SCOPE

| Feature | Description |
|---|---|
| Yawning Detection | Use mouth landmark EAR to detect yawning |
| Head Pose Detection | Alert when head tilts/droops (nodding off) |
| SMS Alert | Send Twilio SMS to emergency contact |
| Email Alert | Send Gmail alert with screenshot |
| Mobile App | Android app with phone camera |
| Raspberry Pi | Embed system in actual car dashboard |
| Deep Learning | Train CNN for even better accuracy |
| YOLO Integration | Use YOLO to detect multiple drivers |

---

### 16. ADVANCED FEATURES – OPTIONAL CODE

#### A) Yawning Detection (Mouth EAR)

```python
# Mouth landmark indices for yawn detection
MOUTH_IDX = [61, 291, 0, 17, 269, 405]

def mouth_aspect_ratio(mouth_points):
    A = distance.euclidean(mouth_points[1], mouth_points[5])
    B = distance.euclidean(mouth_points[2], mouth_points[4])
    C = distance.euclidean(mouth_points[0], mouth_points[3])
    mar = (A + B) / (2.0 * C)
    return mar

# In main loop:
# if mar > 0.6: print("YAWNING DETECTED")
```

#### B) SMS Alert via Twilio

```python
# pip install twilio
from twilio.rest import Client

def send_sms_alert():
    client = Client("YOUR_ACCOUNT_SID", "YOUR_AUTH_TOKEN")
    message = client.messages.create(
        body="⚠️ ALERT: Driver is drowsy! Please check immediately.",
        from_="+1XXXXXXXXXX",   # Your Twilio number
        to="+91XXXXXXXXXX"      # Emergency contact number
    )
    print("SMS sent:", message.sid)
```

#### C) Email Alert via Gmail

```python
import smtplib
from email.mime.text import MIMEText

def send_email_alert():
    msg = MIMEText("Driver drowsiness detected! Immediate attention needed.")
    msg['Subject'] = "⚠️ Drowsiness Alert"
    msg['From']    = "your_email@gmail.com"
    msg['To']      = "emergency@gmail.com"

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login("your_email@gmail.com", "your_app_password")
        server.send_message(msg)
```

#### D) Head Pose / Nodding Detection

```python
# Use MediaPipe landmarks to compute head tilt angle
# If chin-to-forehead angle drops → head is nodding forward
import cv2
import numpy as np

def get_head_tilt_angle(nose_tip, chin):
    dx = chin[0] - nose_tip[0]
    dy = chin[1] - nose_tip[1]
    angle = np.degrees(np.arctan2(dy, dx))
    return angle
# If angle > threshold → alert
```

---

### 17. DEPLOYMENT ON RASPBERRY PI

```bash
# On Raspberry Pi OS (64-bit)
sudo apt update && sudo apt install python3-opencv python3-pip -y
pip3 install mediapipe-rpi numpy pygame scipy imutils
python3 generate_alarm.py
python3 main.py
```

Connect a USB webcam or Raspberry Pi Camera Module.
Connect a buzzer to GPIO pin for hardware alarm.

---

### 18. VIVA QUESTIONS & ANSWERS

**Q1. What is Eye Aspect Ratio?**
A: EAR is a numerical ratio that measures how open an eye is, computed using 6 facial landmark points. Formula: EAR = (||p2-p6|| + ||p3-p5||) / (2×||p1-p4||). Open eye ≈ 0.3, closed ≈ 0.

**Q2. Why do we use 20 consecutive frames and not just 1 frame?**
A: A single blink lasts only 3-5 frames. Using 20 frames (≈0.67 sec) ensures we don't confuse a normal blink with actual drowsiness.

**Q3. What is MediaPipe?**
A: MediaPipe is a Google framework that uses pre-trained deep learning models to detect 468 facial landmarks in real time. It works on CPU/GPU and is optimised for mobile and embedded devices.

**Q4. What is the EAR threshold and how was it determined?**
A: 0.25 is widely used in literature (from the original EAR paper by Soukupová & Čech, 2016). Below 0.25, the eye is considered closed. It can be adjusted for different users.

**Q5. What are the limitations of this system?**
A: Sunglasses block eye detection; low light reduces accuracy; extreme head angles affect landmark detection; no vehicle integration.

**Q6. What is OpenCV?**
A: Open Source Computer Vision Library – a free Python/C++ library for image and video processing, used for webcam access, drawing on frames, and image manipulation.

**Q7. How is this different from a camera recording?**
A: We process each frame in real time (30 FPS), extract mathematical features (EAR), apply logic, and respond in under 33 milliseconds per frame – that's what makes it "real-time."

**Q8. Can this work without internet?**
A: Yes, completely. MediaPipe's model runs locally. No cloud service is used.

**Q9. What is Pygame used for?**
A: Pygame is a Python library for multimedia (games, sound). We use only its `mixer` module to play the alarm WAV sound file.

**Q10. How could you improve this system?**
A: Add yawning detection (mouth EAR), head pose estimation, YOLO for better face detection, deep learning eye classifier, SMS alerts, and Raspberry Pi hardware deployment.

---

### 19. RESUME DESCRIPTION

```
Driver Drowsiness Detection System | Python, OpenCV, MediaPipe, NumPy | [Year]
• Developed a real-time computer vision system to detect driver drowsiness
  using webcam input and Eye Aspect Ratio (EAR) calculation.
• Integrated Google's MediaPipe Face Mesh for 468-point facial landmark
  detection at 30 FPS on standard laptop hardware.
• Implemented audio alarm trigger using Pygame when EAR dropped below
  threshold for 20+ consecutive frames (~0.67 seconds).
• Designed a live HUD dashboard showing FPS, EAR, blink count, and
  driver status with colour-coded alerts.
• Technologies: Python 3.10, OpenCV 4.9, MediaPipe 0.10, NumPy, SciPy, Pygame
```

---

### 20. LINKEDIN PROJECT DESCRIPTION

```
🚗 Driver Drowsiness Detection System

Built a real-time AI-powered driver safety system using Python and Computer Vision.
The system monitors the driver's eyes through a webcam, calculates the Eye Aspect
Ratio (EAR) using MediaPipe's 468-point facial landmark model, and triggers an
audio alarm when drowsiness is detected — all running locally at 30 FPS.

Key highlights:
✅ Real-time eye tracking with Google MediaPipe Face Mesh
✅ Mathematical EAR model for drowsiness classification
✅ Live dashboard: FPS, EAR, blink counter, status
✅ Audio alarm system using Pygame
✅ Optimized for low-end laptops (CPU-only)

Tech Stack: Python | OpenCV | MediaPipe | NumPy | SciPy | Pygame
```

---

### 21. PPT SLIDE CONTENT

**Slide 1 – Title**
- Title: Driver Drowsiness Detection System
- Subtitle: A Real-Time AI-Based Safety System
- Author: [Your Name] | [College Name] | [Year]

**Slide 2 – The Problem**
- 20% of road accidents are caused by drowsy drivers
- Equivalent to drunk driving in terms of impairment
- No low-cost real-time solution for ordinary vehicles
- Image: highway accident statistics chart

**Slide 3 – Our Solution**
- Real-time eye monitoring via webcam
- AI facial landmark detection
- Mathematical EAR-based classification
- Instant audio alarm on drowsiness

**Slide 4 – Technologies Used**
- Python 3 | OpenCV | MediaPipe | NumPy | Pygame
- Brief one-liner on each technology

**Slide 5 – System Architecture**
- Flow diagram: Webcam → MediaPipe → EAR → Logic → Alarm/Display

**Slide 6 – Eye Aspect Ratio (EAR)**
- Diagram showing 6 eye landmark points
- Formula: EAR = (||p2-p6|| + ||p3-p5||) / (2||p1-p4||)
- Graph: EAR value over time (spike down during blink)

**Slide 7 – MediaPipe Face Mesh**
- What it is (Google AI landmark model)
- 468 face points diagram
- Eye points highlighted

**Slide 8 – Algorithm Flow**
- EAR < 0.25 for N consecutive frames → ALARM
- State machine: AWAKE → DROWSY → SLEEPING

**Slide 9 – Live Demo Screenshots**
- Screenshot 1: Normal state (green eye outlines, AWAKE status)
- Screenshot 2: Drowsy state (red outlines)
- Screenshot 3: Alarm state (red banner)

**Slide 10 – Results**
- Works at 25-30 FPS on standard laptop
- Detection latency: ~0.67 seconds
- Alarm triggers reliably in controlled tests

**Slide 11 – Advantages & Limitations**
- Table: Advantages vs Limitations

**Slide 12 – Future Scope**
- Yawning detection
- SMS/Email alerts
- Raspberry Pi embedded system
- Deep learning model

**Slide 13 – Conclusion**
- Low-cost, real-time, webcam-based driver safety system
- Demonstrates practical AI/CV applications
- Scalable to advanced hardware

**Slide 14 – References**
- Soukupová & Čech (2016) – Real-Time Eye Blink Detection using Facial Landmarks
- MediaPipe Documentation – Google
- OpenCV Official Documentation

---

### 22. CONCLUSION

This Driver Drowsiness Detection System demonstrates how modern AI and Computer Vision tools can be combined to solve a real-world safety problem. Using only a webcam, a Python environment, and free open-source libraries, we built a system that:

- Tracks the driver's eyes 30 times per second
- Calculates a mathematical measure of eye openness (EAR)
- Automatically triggers an alarm when drowsiness is detected

The project is a practical example of applying AI not just in theory, but in a meaningful, potentially life-saving application. With further development — adding SMS alerts, Raspberry Pi hardware, or deep learning — this system could realistically be deployed in actual vehicles.

---

*Made with ❤️ | Python + OpenCV + MediaPipe*
