import cv2
import mediapipe as mp
import pyautogui
import math
import time
import os
import sys

# ---------- Detect available cameras ----------
def find_cameras(max_check=5):
    available = []
    for i in range(max_check):
        test = cv2.VideoCapture(i)
        if test.isOpened():
            available.append(i)
            test.release()
    return available


cameras = find_cameras()
if not cameras:
    print("ERROR: No cameras found!")
    input("Press Enter to exit...")
    sys.exit(1)

if len(cameras) == 1:
    CAM_INDEX = cameras[0]
    print(f"Using camera {CAM_INDEX}")
else:
    print("\nAvailable cameras:")
    for i, cam in enumerate(cameras):
        print(f"  [{cam}] Camera {cam}")
    while True:
        choice = input(f"\nSelect camera ({', '.join(str(c) for c in cameras)}): ").strip()
        if choice.isdigit() and int(choice) in cameras:
            CAM_INDEX = int(choice)
            break
        print("Invalid choice, try again.")

# ---------- Settings ----------
CAM_WIDTH = 640
CAM_HEIGHT = 480
SMOOTHING = 5          # higher = smoother but slower cursor
CLICK_COOLDOWN = 0.4   # seconds between clicks
PUSH_THRESHOLD = -0.07 # z-depth threshold: finger "pushed" toward camera (more negative = closer)
SCROLL_SPEED = 2       # scroll amount per frame while in scroll mode
FRAME_MARGIN = 80      # ignore hand positions near the edge of the frame

# Disable PyAutoGUI fail-safe pause for responsiveness
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = True  # move mouse to top-left corner to abort

# ---------- Resolve model path ----------
if getattr(sys, 'frozen', False):
    base_dir = sys._MEIPASS
else:
    base_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(base_dir, "hand_landmarker.task")

# ---------- Init MediaPipe Hand Landmarker (new tasks API) ----------
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7,
)

landmarker = HandLandmarker.create_from_options(options)

# ---------- Init webcam ----------
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

screen_w, screen_h = pyautogui.size()

# Smoothing buffers
prev_x, prev_y = 0, 0
last_click_time = 0
last_right_click_time = 0
scroll_y_start = None
frame_timestamp = 0
index_was_pushed = False
middle_was_pushed = False

# Hand connections for drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


print("Hand Mouse Control - press 'q' in the preview window to quit.")
print("Move mouse to the top-left corner of your screen to emergency-stop.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror the frame so movement feels natural
    frame = cv2.flip(frame, 1)
    frame_h, frame_w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    frame_timestamp += 33  # ~30fps in milliseconds
    result = landmarker.detect_for_video(mp_image, frame_timestamp)

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]

        # Draw landmarks
        for lm in hand:
            cx, cy = int(lm.x * frame_w), int(lm.y * frame_h)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), cv2.FILLED)

        # Draw connections
        for start_idx, end_idx in HAND_CONNECTIONS:
            p1 = (int(hand[start_idx].x * frame_w), int(hand[start_idx].y * frame_h))
            p2 = (int(hand[end_idx].x * frame_w), int(hand[end_idx].y * frame_h))
            cv2.line(frame, p1, p2, (0, 255, 0), 2)

        # Get fingertip positions in pixels
        index_tip = (int(hand[8].x * frame_w), int(hand[8].y * frame_h))
        middle_tip = (int(hand[12].x * frame_w), int(hand[12].y * frame_h))

        # Depth: z relative to wrist (landmark 0). More negative = closer to camera.
        wrist_z = hand[0].z
        index_z = hand[8].z - wrist_z
        middle_z = hand[12].z - wrist_z

        index_pushed = index_z < PUSH_THRESHOLD
        middle_pushed = middle_z < PUSH_THRESHOLD

        # Map index finger position to screen, clamping the usable area
        margin = FRAME_MARGIN
        usable_w = frame_w - 2 * margin
        usable_h = frame_h - 2 * margin

        raw_x = (index_tip[0] - margin) / usable_w
        raw_y = (index_tip[1] - margin) / usable_h
        raw_x = max(0.0, min(1.0, raw_x))
        raw_y = max(0.0, min(1.0, raw_y))

        target_x = raw_x * screen_w
        target_y = raw_y * screen_h

        any_push = index_pushed or middle_pushed

        # Only move cursor when no finger is pushed toward camera
        if not any_push:
            prev_x += (target_x - prev_x) / SMOOTHING
            prev_y += (target_y - prev_y) / SMOOTHING
            pyautogui.moveTo(int(prev_x), int(prev_y))

        # Highlight fingertips — color changes when pushed
        idx_color = (0, 0, 255) if index_pushed else (255, 0, 255)
        mid_color = (0, 0, 255) if middle_pushed else (255, 165, 0)
        cv2.circle(frame, index_tip, 10, idx_color, cv2.FILLED)
        cv2.circle(frame, middle_tip, 10, mid_color, cv2.FILLED)

        # Show depth values for debugging
        cv2.putText(frame, f"Index Z: {index_z:.3f}", (10, frame_h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Middle Z: {middle_z:.3f}", (10, frame_h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Scroll: both index + middle pushed toward camera
        if index_pushed and middle_pushed:
            if scroll_y_start is None:
                scroll_y_start = index_tip[1]
            else:
                delta = index_tip[1] - scroll_y_start
                if abs(delta) > 10:
                    pyautogui.scroll(-int(delta / 20 * SCROLL_SPEED))
                    scroll_y_start = index_tip[1]
            cv2.putText(frame, "SCROLL", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
        else:
            scroll_y_start = None

            # Left click: only index pushed (on push-down, not held)
            if index_pushed and not index_was_pushed:
                now = time.time()
                if now - last_click_time > CLICK_COOLDOWN:
                    pyautogui.click()
                    last_click_time = now
                    cv2.putText(frame, "LEFT CLICK", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # Right click: only middle pushed (on push-down, not held)
            if middle_pushed and not middle_was_pushed:
                now = time.time()
                if now - last_right_click_time > CLICK_COOLDOWN:
                    pyautogui.rightClick()
                    last_right_click_time = now
                    cv2.putText(frame, "RIGHT CLICK", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        index_was_pushed = index_pushed
        middle_was_pushed = middle_pushed

    cv2.imshow("Hand Mouse Control", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

landmarker.close()
cap.release()
cv2.destroyAllWindows()
