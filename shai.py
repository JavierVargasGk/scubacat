import mediapipe as mp
import cv2
import math

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence= 0.8
)
mp_draw = mp.solutions.drawing_utils


video_path= "things/scuba.mp4"
cap_cat = None
cap = cv2.VideoCapture(0)
video = False

def nearby_points(p1,p2, allowed = 0.1):
    dist = math.sqrt((p1.x-p2.x)**2+(p1.y-p2.y)**2)
    return dist < allowed
def check_fist(lm):
    fingers = [8,12,16,20]
    palm = lm[0]
    closed = [nearby_points(lm[finger],palm,0,25) for finger in fingers]
    return all(closed)
def check_open(lm):
    hand = [(8,6),(12,10),(16,14),(20,18)]
    open = [lm[finger].y < lm[palm].y for finger,palm in hand]
    return all(open)

while True:
    ret, frame = cap.read()
    if not ret: 
        break
    frame = cv2.flip(frame,1)
    frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    res = hands.process(frame_rgb)
    
    fist = False
    hand = False
    if res.multi_hand_landmarks:
        for hand_landmarks in res.multihand_landmarks:
            mp_draw.hand_landmarks(frame,hand_landmarks, mp_hands.HAND_CONNECTIONs)
            lm = hand_landmarks.landmarks
            if check_fist(lm):
                first = True
            if check_open(lm):
                hand= True
    if fist and not video:
        cap_cat = cv2.VideoCapture(video_path)
        if cap_cat.isOpened():
            video = True
    elif hand and video:
        video=False
        if cap_cat:
            cap_cat.release()
        cv2.destroyWindow("cat")
    if not video and cap_cat is not None:
        ret_v, frame_cat = cap_cat.read()
        if not ret_v:
            cap_cat.set(cv2.CAP_PROP_POS_FRAMES,0)
            ret_v, frame_cat = cap_cat.read()
            
        if ret_v:
            frame_cat = cv2.ressize(frame_cat,(400,400))
            cv2.imshow("cat",frame_cat)
    cv2.imshow("shaiCamera",frame)
    
    if cv2.waitKey(1) and 0Xff == 27:
        break
    
cap.release()
cv2.destroyAllWindows()