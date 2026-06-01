import cv2
import mediapipe as mp
from math import hypot
from playsound import playsound
import threading

print("Program Started")

# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Eye landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Open webcam
cap = cv2.VideoCapture(0)
print("Camera Opened:", cap.isOpened())

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

closed_frames = 0
THRESHOLD = 0.22
FRAME_LIMIT = 20
alarm_playing = False

def play_alarm():
    global alarm_playing
    alarm_playing = True
    try:
        playsound("alarm.wav")  # Ensure alarm.wav is in the same folder
    except Exception as e:
        print("Alarm Error:", e)
    alarm_playing = False

def eye_aspect_ratio(landmarks, eye_points, w, h):
    points = []
    for p in eye_points:
        x = int(landmarks[p].x * w)
        y = int(landmarks[p].y * h)
        points.append((x, y))

    vertical1 = hypot(points[1][0] - points[5][0], points[1][1] - points[5][1])
    vertical2 = hypot(points[2][0] - points[4][0], points[2][1] - points[4][1])
    horizontal = hypot(points[0][0] - points[3][0], points[0][1] - points[3][1])

    # Prevent division by zero just in case
    if horizontal == 0:
        return 0
        
    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    h, w, _ = frame.shape

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
            ear = (left_ear + right_ear) / 2

            cv2.putText(frame, f"EAR: {ear:.2f}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if ear < THRESHOLD:
                closed_frames += 1
                cv2.putText(frame, "DROWSINESS ALERT!", (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                if closed_frames > FRAME_LIMIT:
                    if not alarm_playing:
                        threading.Thread(target=play_alarm, daemon=True).start()
            else:
                closed_frames = 0

    cv2.imshow("Driver Drowsiness Detection", frame)

    # Press 'ESC' to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("Program Ended")
