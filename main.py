"""
============================================================
  DRIVER DROWSINESS DETECTION SYSTEM
  Built with: Python | OpenCV | MediaPipe | NumPy | Pygame
  Author: College Mini Project
  Description: Detects driver drowsiness in real-time using
               Eye Aspect Ratio (EAR) via facial landmarks.
============================================================
"""

# ─── STEP 1: IMPORT REQUIRED LIBRARIES ───────────────────────────────────────
import cv2                          # OpenCV: for webcam access and image processing
import mediapipe as mp             # MediaPipe: for facial landmark detection
import numpy as np                 # NumPy: for mathematical calculations
import pygame                      # Pygame: for playing the alarm sound
import time                        # time: for tracking elapsed seconds
import os                          # os: for file path handling
import sys                         # sys: for graceful exit
from imutils import face_utils     # imutils: utility functions (optional, for helpers)
from scipy.spatial import distance # scipy: for computing Euclidean distances

# ─── STEP 2: CONFIGURATION / TUNABLE PARAMETERS ──────────────────────────────
# ▸ EAR threshold: if EAR drops below this → eye is CLOSED
EAR_THRESHOLD       = 0.25

# ▸ How many CONSECUTIVE frames must have EAR < threshold before alarm fires
EAR_CONSEC_FRAMES   = 20           # ~0.67 sec at 30 FPS

# ▸ How many closed-eye frames count as one "blink"
BLINK_CONSEC_FRAMES = 3

# ▸ Path to the alarm sound file (place alarm.wav in same folder as main.py)
ALARM_PATH = os.path.join(os.path.dirname(__file__), "alarm.wav")

# ─── STEP 3: MediaPipe FACE MESH LANDMARK INDICES ────────────────────────────
# MediaPipe's 468-point face mesh model provides coordinates for every
# facial feature. We only need the 6 points around each eye.
#
# LEFT EYE  landmark indices (from MediaPipe face-mesh topology):
LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]
# RIGHT EYE landmark indices:
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# ─── STEP 4: HELPER – COMPUTE EAR (EYE ASPECT RATIO) ────────────────────────
def eye_aspect_ratio(eye_points):
    """
    Calculate the Eye Aspect Ratio (EAR) for one eye.

    FORMULA:
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    where p1..p6 are the 6 landmark points of the eye:
        p1 ─────── p4     (horizontal endpoints)
          p2     p5
          p3     p6       (vertical pairs)

    When the eye is OPEN  → EAR ≈ 0.25-0.40
    When the eye is CLOSED → EAR ≈ 0.0

    Parameters
    ----------
    eye_points : list of (x, y) tuples – 6 landmark coordinates for one eye

    Returns
    -------
    ear : float – Eye Aspect Ratio value
    """
    # Vertical distances: distance between the upper and lower eyelid
    A = distance.euclidean(eye_points[1], eye_points[5])  # p2 - p6
    B = distance.euclidean(eye_points[2], eye_points[4])  # p3 - p5

    # Horizontal distance: corner-to-corner width of the eye
    C = distance.euclidean(eye_points[0], eye_points[3])  # p1 - p4

    # EAR formula
    ear = (A + B) / (2.0 * C)
    return ear


# ─── STEP 5: ALARM SYSTEM ─────────────────────────────────────────────────────
class AlarmSystem:
    """
    Manages the audio alarm using the Pygame library.
    Plays alarm.wav when drowsiness is detected, stops it otherwise.
    """

    def __init__(self, alarm_path):
        """Set up the Pygame mixer and load the alarm sound."""
        pygame.mixer.init()
        self.alarm_path  = alarm_path
        self.alarm_on    = False
        self.sound_loaded = False

        # Try to load the alarm file; gracefully handle if it's missing
        if os.path.exists(alarm_path):
            try:
                pygame.mixer.music.load(alarm_path)
                self.sound_loaded = True
                print("[INFO] Alarm sound loaded successfully.")
            except Exception as e:
                print(f"[WARNING] Could not load alarm: {e}")
        else:
            print(f"[WARNING] alarm.wav not found at: {alarm_path}")
            print("[WARNING] Run generate_alarm.py first to create it.")

    def play(self):
        """Start playing the alarm (loops continuously)."""
        if self.sound_loaded and not self.alarm_on:
            pygame.mixer.music.play(-1)   # -1 means loop forever
            self.alarm_on = True

    def stop(self):
        """Stop the alarm."""
        if self.alarm_on:
            pygame.mixer.music.stop()
            self.alarm_on = False


# ─── STEP 6: UI DRAWING HELPERS ──────────────────────────────────────────────
def draw_rounded_rect(img, pt1, pt2, color, radius=10, thickness=-1):
    """Draw a filled rectangle with rounded corners on the frame."""
    x1, y1 = pt1
    x2, y2 = pt2
    overlay = img.copy()
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.circle(overlay, (x1 + radius, y1 + radius), radius, color, thickness)
    cv2.circle(overlay, (x2 - radius, y1 + radius), radius, color, thickness)
    cv2.circle(overlay, (x1 + radius, y2 - radius), radius, color, thickness)
    cv2.circle(overlay, (x2 - radius, y2 - radius), radius, color, thickness)
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)


def draw_dashboard(frame, fps, blink_count, ear, consec_frames, alarm_on, status):
    """
    Draw a clean HUD (Heads-Up Display) panel on the video frame.

    Shows: FPS | EAR | Blink Count | Status | Warning banner
    """
    h, w = frame.shape[:2]

    # ── Semi-transparent side panel (right side) ──────────────────────────
    panel_x = w - 230
    overlay  = frame.copy()
    cv2.rectangle(overlay, (panel_x - 10, 10), (w - 10, 200), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # ── Panel Title ───────────────────────────────────────────────────────
    cv2.putText(frame, "DRIVER MONITOR", (panel_x, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1, cv2.LINE_AA)
    cv2.line(frame, (panel_x - 8, 42), (w - 12, 42), (0, 200, 255), 1)

    # ── Stats ─────────────────────────────────────────────────────────────
    cv2.putText(frame, f"FPS    : {fps:.1f}",      (panel_x, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"EAR    : {ear:.3f}",      (panel_x, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"BLINKS : {blink_count}",  (panel_x, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"FRAMES : {consec_frames}",  (panel_x, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)

    # ── Eye Status ────────────────────────────────────────────────────────
    status_color = (0, 255, 0) if status == "AWAKE" else (0, 165, 255) if status == "DROWSY" else (0, 0, 255)
    cv2.putText(frame, f"STATUS : {status}", (panel_x, 168),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2, cv2.LINE_AA)

    # ── Alarm / Warning Banner (bottom) ──────────────────────────────────
    if alarm_on:
        banner_overlay = frame.copy()
        cv2.rectangle(banner_overlay, (0, h - 70), (w, h), (0, 0, 180), -1)
        cv2.addWeighted(banner_overlay, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, "⚠  DROWSINESS ALERT! PULL OVER NOW  ⚠",
                    (int(w * 0.04), h - 28),
                    cv2.FONT_HERSHEY_DUPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)

    # ── EAR Threshold Bar (left side) ────────────────────────────────────
    bar_x, bar_y, bar_h = 20, 60, 150
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + 16, bar_y + bar_h), (60, 60, 60), -1)
    fill = int(bar_h * min(ear / 0.45, 1.0))
    bar_color = (0, 255, 0) if ear >= EAR_THRESHOLD else (0, 0, 255)
    cv2.rectangle(frame, (bar_x, bar_y + bar_h - fill),
                  (bar_x + 16, bar_y + bar_h), bar_color, -1)
    threshold_y = bar_y + bar_h - int(bar_h * EAR_THRESHOLD / 0.45)
    cv2.line(frame, (bar_x - 4, threshold_y), (bar_x + 20, threshold_y), (255, 255, 0), 1)
    cv2.putText(frame, "EAR", (bar_x - 4, bar_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)


def draw_eye_contour(frame, eye_points, color):
    """Draw a polygon around the detected eye landmarks."""
    pts = np.array(eye_points, dtype=np.int32)
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=1)
    for (x, y) in eye_points:
        cv2.circle(frame, (x, y), 2, color, -1)


# ─── STEP 7: MAIN PROGRAM ────────────────────────────────────────────────────
def main():
    """
    Main function – Entry point of the Drowsiness Detection System.

    Flow:
      1. Initialize MediaPipe Face Mesh
      2. Open webcam
      3. For each frame:
          a. Detect face landmarks
          b. Extract eye points
          c. Compute EAR
          d. Count consecutive low-EAR frames
          e. Trigger/stop alarm
          f. Draw HUD and display frame
    """

    print("=" * 60)
    print("  DRIVER DROWSINESS DETECTION SYSTEM  ")
    print("  Press [Q] to quit")
    print("=" * 60)

    # ── Init alarm ────────────────────────────────────────────────────────
    alarm = AlarmSystem(ALARM_PATH)

    # ── Init MediaPipe Face Mesh ──────────────────────────────────────────
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh    = mp_face_mesh.FaceMesh(
        max_num_faces       = 1,        # Only detect 1 face (the driver)
        refine_landmarks    = True,     # More precise eye landmarks
        min_detection_confidence = 0.5,
        min_tracking_confidence  = 0.5
    )

    # ── Open Webcam ───────────────────────────────────────────────────────
    print("[INFO] Starting webcam...")
    cap = cv2.VideoCapture(0)           # 0 = default/built-in webcam

    if not cap.isOpened():
        print("[ERROR] Cannot access webcam. Check if it is connected.")
        sys.exit(1)

    # Optional: lower resolution for better performance on slow laptops
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # ── State Variables ───────────────────────────────────────────────────
    consec_frames = 0       # How many consecutive frames had EAR < threshold
    blink_count   = 0       # Total eye blinks detected
    alarm_on      = False   # Is the alarm currently firing?
    status        = "AWAKE" # Current status label shown on screen

    # FPS tracking
    fps_counter  = 0
    fps          = 0.0
    fps_time     = time.time()

    # ── MAIN LOOP ─────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()         # Read one frame from webcam
        if not ret:
            print("[ERROR] Failed to read frame from webcam.")
            break

        # Mirror the frame so it feels like a selfie-camera
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        # Convert BGR (OpenCV default) → RGB (MediaPipe requirement)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── Face Landmark Detection ───────────────────────────────────────
        result = face_mesh.process(rgb_frame)

        ear = 0.0  # Default EAR when no face is detected

        if result.multi_face_landmarks:
            # We only care about the first (and only) detected face
            face_landmarks = result.multi_face_landmarks[0]

            # Extract (x, y) pixel coordinates for each eye landmark
            left_eye_pts  = []
            right_eye_pts = []

            for idx in LEFT_EYE_IDX:
                lm = face_landmarks.landmark[idx]
                left_eye_pts.append((int(lm.x * w), int(lm.y * h)))

            for idx in RIGHT_EYE_IDX:
                lm = face_landmarks.landmark[idx]
                right_eye_pts.append((int(lm.x * w), int(lm.y * h)))

            # ── Compute EAR for both eyes, then average ───────────────────
            left_ear  = eye_aspect_ratio(left_eye_pts)
            right_ear = eye_aspect_ratio(right_eye_pts)
            ear       = (left_ear + right_ear) / 2.0

            # ── Draw eye contours ─────────────────────────────────────────
            eye_color = (0, 255, 0) if ear >= EAR_THRESHOLD else (0, 0, 255)
            draw_eye_contour(frame, left_eye_pts,  eye_color)
            draw_eye_contour(frame, right_eye_pts, eye_color)

            # ── Drowsiness Logic ──────────────────────────────────────────
            if ear < EAR_THRESHOLD:
                # Eye is CLOSED this frame → increment counter
                consec_frames += 1

                if consec_frames < EAR_CONSEC_FRAMES:
                    status = "DROWSY"
                else:
                    # Eyes have been closed long enough → SLEEPING!
                    status   = "SLEEPING"
                    alarm_on = True
                    alarm.play()

            else:
                # Eye is OPEN → reset consecutive counter
                if BLINK_CONSEC_FRAMES <= consec_frames < EAR_CONSEC_FRAMES:
                    blink_count += 1   # Count it as a complete blink

                consec_frames = 0
                status        = "AWAKE"
                alarm_on      = False
                alarm.stop()

        else:
            # No face detected in this frame
            status = "NO FACE"

        # ── FPS Calculation ───────────────────────────────────────────────
        fps_counter += 1
        elapsed = time.time() - fps_time
        if elapsed >= 1.0:
            fps        = fps_counter / elapsed
            fps_counter = 0
            fps_time   = time.time()

        # ── Draw the Dashboard HUD ────────────────────────────────────────
        draw_dashboard(frame, fps, blink_count, ear, consec_frames, alarm_on, status)

        # ── Show the Frame ────────────────────────────────────────────────
        cv2.imshow("Driver Drowsiness Detection System", frame)

        # ── Key Press Handler ─────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            print("[INFO] Quit key pressed. Exiting...")
            break
        elif key == ord('r') or key == ord('R'):
            # R = Reset blink counter
            blink_count = 0
            print("[INFO] Blink counter reset.")

    # ── Cleanup ───────────────────────────────────────────────────────────
    print("[INFO] Releasing resources...")
    alarm.stop()
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    pygame.mixer.quit()
    print("[INFO] System shut down cleanly. Goodbye!")


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
