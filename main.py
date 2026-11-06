import cv2
import mediapipe as mp
import numpy as np

def neon(frame, shift):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    banded = ((gray.astype(np.uint16) * 3 + shift) % 256).astype(np.uint8)
    coloured = cv2.applyColorMap(banded, cv2.COLORMAP_HSV)
    edges = cv2.Canny(gray, 40, 110)
    coloured[edges > 0] = (0, 0, 0)
    return coloured

def neon_hand_frame():
    cam = cv2.VideoCapture(1)
    mp_hands = mp.solutions.hands.Hands(max_num_hands=2)
    mp_drawing = mp.solutions.drawing_utils
    shift = 0

    while cam.isOpened():
        success, frame = cam.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(frame_rgb)

        points = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for tip in (4, 8):
                    lm = hand_landmarks.landmark[tip]
                    x = int(lm.x * frame.shape[1])
                    y = int(lm.y * frame.shape[0])
                    points.append([x, y])

            if len(points) == 4:
                # Both hands found
                points_array = np.array(points, dtype=np.int32)
                hull = cv2.convexHull(points_array)

                # Mask: white inside the shape, black everywhere else
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.fillConvexPoly(mask, hull, 255)

                # Copy neon pixels into frame, only where mask is white
                neon_frame = neon(frame, shift)
                frame[mask == 255] = neon_frame[mask == 255]

        shift = (shift + 4) % 256

        cv2.imshow("Neon Hand Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

neon_hand_frame()