import cv2
import math
import time
import mediapipe as mp
from ffpyplayer.player import MediaPlayer

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8,
)
mp_draw = mp.solutions.drawing_utils

video_path = "things/scuba.mp4"
cap_cat = None
audio_cat = None
cap = cv2.VideoCapture(0)
video = False

hand_position_history = []
HISTORY_FRAMES = 2
MOTION_THRESHOLD = 0.04
COOLDOWN_DURATION = 1.5

last_movement_time = 0.0


def nearby_points(p1, p2, allowed=0.1):
    dist = math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)
    return dist < allowed


def check_fist(lm):
    fingers = [8, 12, 16, 20]
    palm = lm[0]
    closed = [nearby_points(lm[finger], palm, 0.25) for finger in fingers]
    return all(closed)


def check_open(lm):
    hand_pairs = [(8, 6), (12, 10), (16, 14), (20, 18)]
    open_fingers = [lm[finger].y < lm[palm].y for finger, palm in hand_pairs]
    return all(open_fingers)


def check_movement(current_palm_node):
    global hand_position_history

    current_pos = [current_palm_node.x, current_palm_node.y]
    hand_position_history.append(current_pos)

    if len(hand_position_history) > HISTORY_FRAMES:
        hand_position_history.pop(0)

    if len(hand_position_history) < HISTORY_FRAMES:
        return False

    total_distance = 0.0
    for i in range(len(hand_position_history) - 1):
        p1 = hand_position_history[i]
        p2 = hand_position_history[i + 1]
        total_distance += math.sqrt(
            (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
        )

    return total_distance > MOTION_THRESHOLD


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(frame_rgb)

    force_shutdown = False

    if res.multi_hand_landmarks:
        for hand_landmarks in res.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

        if len(res.multi_hand_landmarks) == 2:
            lm_hand1 = res.multi_hand_landmarks[0].landmark
            lm_hand2 = res.multi_hand_landmarks[1].landmark

            hand1_is_fist = check_fist(lm_hand1)
            hand1_is_open = check_open(lm_hand1)
            hand2_is_fist = check_fist(lm_hand2)
            hand2_is_open = check_open(lm_hand2)

            if hand1_is_fist and hand2_is_open:
                if check_movement(lm_hand2[0]):
                    last_movement_time = time.time()
            elif hand2_is_fist and hand1_is_open:
                if check_movement(lm_hand1[0]):
                    last_movement_time = time.time()
        else:
            hand_position_history.clear()
    else:
        hand_position_history.clear()
        force_shutdown = True

    time_since_last_move = time.time() - last_movement_time
    timer_active = time_since_last_move < COOLDOWN_DURATION

    if timer_active and not force_shutdown and not video:
        cap_cat = cv2.VideoCapture(video_path)
        audio_cat = MediaPlayer(video_path)
        if cap_cat.isOpened():
            video = True
    elif (force_shutdown or not timer_active) and video:
        video = False
        if cap_cat:
            cap_cat.release()
            cap_cat = None
        if audio_cat:
            audio_cat.close_player()
            audio_cat = None
        try:
            cv2.destroyWindow("cat")
        except cv2.error:
            pass

    if video and cap_cat is not None:
        ret_v, frame_cat = cap_cat.read()
        audio_frame, val = audio_cat.get_frame() if audio_cat else (None, 0)

        if not ret_v:
            cap_cat.set(cv2.CAP_PROP_POS_FRAMES, 0)
            if audio_cat:
                audio_cat.seek(0, relative=False)
            ret_v, frame_cat = cap_cat.read()

        if ret_v:
            frame_cat = cv2.resize(frame_cat, (400, 400))
            cv2.imshow("cat", frame_cat)

        if val == "paused":
            time.sleep(0.01)
        elif val > 0:
            time.sleep(val)

    cv2.imshow("shaiCamera", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
if cap_cat:
    cap_cat.release()
if audio_cat:
    audio_cat.close_player()
cv2.destroyAllWindows()