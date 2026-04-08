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
CURL_RATIO = 0.65      # finger is "curled" when length drops below this fraction of resting length
SCROLL_SPEED = 200     # scroll amount per frame while in scroll mode
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

# State
prev_x, prev_y = 0, 0
last_click_time = 0
last_right_click_time = 0
last_dbl_click_time = 0
pinch_was_active = False
PINCH_DIST = 40        # pixel distance between thumb & index for double click
scroll_y_start = None
frame_timestamp = 0
index_was_curled = False
middle_was_curled = False
was_scrolling = False  # track if we were just scrolling
scroll_grace_time = 0
scroll_locked = False
SCROLL_GRACE = 0.3


# Hand connections for drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def dist2d(a, b, fw, fh):
    """2D pixel distance between two landmarks."""
    return math.hypot((a.x - b.x) * fw, (a.y - b.y) * fh)


def finger_length(hand, tip_idx, base_idx, fw, fh):
    """Distance from fingertip to finger base (MCP joint) in pixels."""
    return dist2d(hand[tip_idx], hand[base_idx], fw, fh)


# ---------- Calibration ----------
CALIBRATION_SECONDS = 3
CALIBRATION_FRAMES = int(CALIBRATION_SECONDS * 30)

print("Hand Mouse Control - press 'q' in the preview window to quit.")
print("Move mouse to the top-left corner of your screen to emergency-stop.")
print(f"\n>>> Hold your hand OPEN in front of the camera for {CALIBRATION_SECONDS}s to calibrate...")

index_len_samples = []
middle_len_samples = []
calibrated = False
cal_frame_count = 0

while cap.isOpened() and not calibrated:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_h, frame_w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    frame_timestamp += 33
    result = landmarker.detect_for_video(mp_image, frame_timestamp)

    # Show calibration UI
    progress = min(1.0, cal_frame_count / CALIBRATION_FRAMES) if index_len_samples else 0.0
    bar_w = int(400 * progress)
    cv2.rectangle(frame, (120, 220), (520, 260), (50, 50, 50), -1)
    cv2.rectangle(frame, (120, 220), (120 + bar_w, 260), (0, 255, 0), -1)
    cv2.putText(frame, f"Calibrating... {int(progress * 100)}%", (160, 210),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    if not index_len_samples:
        cv2.putText(frame, "Show your OPEN hand to the camera", (80, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]

        # Draw hand during calibration
        for lm in hand:
            cx, cy = int(lm.x * frame_w), int(lm.y * frame_h)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), cv2.FILLED)
        for start_idx, end_idx in HAND_CONNECTIONS:
            p1 = (int(hand[start_idx].x * frame_w), int(hand[start_idx].y * frame_h))
            p2 = (int(hand[end_idx].x * frame_w), int(hand[end_idx].y * frame_h))
            cv2.line(frame, p1, p2, (0, 255, 0), 2)

        # Measure resting (open) finger lengths
        # Index: tip=8, base MCP=5
        # Middle: tip=12, base MCP=9
        index_len_samples.append(finger_length(hand, 8, 5, frame_w, frame_h))
        middle_len_samples.append(finger_length(hand, 12, 9, frame_w, frame_h))
        cal_frame_count += 1

        if cal_frame_count >= CALIBRATION_FRAMES:
            index_rest_len = sum(index_len_samples) / len(index_len_samples)
            middle_rest_len = sum(middle_len_samples) / len(middle_len_samples)
            # Curl thresholds: finger is "curled" when shorter than this
            index_curl_threshold = index_rest_len * CURL_RATIO
            middle_curl_threshold = middle_rest_len * CURL_RATIO
            calibrated = True
            print(f"Calibrated! Index resting: {index_rest_len:.1f}px, Middle resting: {middle_rest_len:.1f}px")
            print(f"Curl thresholds — Index: {index_curl_threshold:.1f}px, Middle: {middle_curl_threshold:.1f}px")

    cv2.imshow("Hand Mouse Control", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        cap.release()
        cv2.destroyAllWindows()
        sys.exit(0)

if not calibrated:
    print("Calibration failed — no hand detected.")
    cap.release()
    cv2.destroyAllWindows()
    sys.exit(1)

# ---------- Main loop ----------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror the frame so movement feels natural
    frame = cv2.flip(frame, 1)
    frame_h, frame_w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    frame_timestamp += 33
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
        thumb_tip = (int(hand[4].x * frame_w), int(hand[4].y * frame_h))

        # Measure current finger lengths
        index_len = finger_length(hand, 8, 5, frame_w, frame_h)
        middle_len = finger_length(hand, 12, 9, frame_w, frame_h)

        index_curled = index_len < index_curl_threshold
        middle_curled = middle_len < middle_curl_threshold

        # Map index finger position to screen
        margin = FRAME_MARGIN
        usable_w = frame_w - 2 * margin
        usable_h = frame_h - 2 * margin

        raw_x = (index_tip[0] - margin) / usable_w
        raw_y = (index_tip[1] - margin) / usable_h
        raw_x = max(0.0, min(1.0, raw_x))
        raw_y = max(0.0, min(1.0, raw_y))

        target_x = raw_x * screen_w
        target_y = raw_y * screen_h

        any_curl = index_curled or middle_curled

        # Only move cursor when no finger is curled
        if not any_curl:
            prev_x += (target_x - prev_x) / SMOOTHING
            prev_y += (target_y - prev_y) / SMOOTHING
            pyautogui.moveTo(int(prev_x), int(prev_y))

        # Highlight fingertips — color changes when curled
        idx_color = (0, 0, 255) if index_curled else (255, 0, 255)
        mid_color = (0, 0, 255) if middle_curled else (255, 165, 0)
        cv2.circle(frame, index_tip, 10, idx_color, cv2.FILLED)
        cv2.circle(frame, middle_tip, 10, mid_color, cv2.FILLED)

        # Show finger length debug info
        cv2.putText(frame, f"Index: {index_len:.0f}/{index_curl_threshold:.0f}px", (10, frame_h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Middle: {middle_len:.0f}/{middle_curl_threshold:.0f}px", (10, frame_h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Scroll: both index + middle curled (with grace period)
        both_curled = index_curled and middle_curled
        now = time.time()

        if both_curled:
            scroll_grace_time = now

        in_scroll_mode = both_curled or (scroll_y_start is not None and now - scroll_grace_time < SCROLL_GRACE)

        if in_scroll_mode:
            # Entered scroll — cancel any pending click
            pending_click = None

            if scroll_y_start is None:
                scroll_y_start = index_tip[1]
                scroll_locked = False
            else:
                delta = index_tip[1] - scroll_y_start
                if not scroll_locked:
                    if abs(delta) > 40:
                        scroll_locked = True
                        scroll_y_start = index_tip[1]
                    cv2.putText(frame, "SCROLL - move up/down", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                else:
                    if abs(delta) > 5:
                        pyautogui.scroll(int(delta * 3))
                        scroll_y_start = index_tip[1]
                    cv2.putText(frame, "SCROLLING", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
        else:
            scroll_y_start = None
            scroll_locked = False

            # Clicks fire on RELEASE (uncurl), and only if we weren't scrolling
            if not was_scrolling:
                # Left click: index was curled, now released, middle wasn't curled
                if index_was_curled and not index_curled and not middle_was_curled:
                    if now - last_click_time > CLICK_COOLDOWN:
                        pyautogui.click()
                        last_click_time = now
                        cv2.putText(frame, "LEFT CLICK", (10, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                # Right click: middle was curled, now released, index wasn't curled
                if middle_was_curled and not middle_curled and not index_was_curled:
                    if now - last_right_click_time > CLICK_COOLDOWN:
                        pyautogui.rightClick()
                        last_right_click_time = now
                        cv2.putText(frame, "RIGHT CLICK", (10, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        was_scrolling = in_scroll_mode

        # Double click: pinch index + thumb together (on release)
        pinch_dist = math.hypot(index_tip[0] - thumb_tip[0], index_tip[1] - thumb_tip[1])
        pinch_active = pinch_dist < PINCH_DIST

        if pinch_was_active and not pinch_active:
            if now - last_dbl_click_time > CLICK_COOLDOWN:
                pyautogui.doubleClick()
                last_dbl_click_time = now
                cv2.putText(frame, "DOUBLE CLICK", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        pinch_was_active = pinch_active

        index_was_curled = index_curled
        middle_was_curled = middle_curled

    cv2.imshow("Hand Mouse Control", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

landmarker.close()
cap.release()
cv2.destroyAllWindows()
