import cv2
import numpy as np
import os

CAM_INDEX = 0
FLIP_MODE = 1

points = []

def mouse_callback(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))
        print(f"📍 Point {len(points)}: {x}, {y}")

cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

cv2.namedWindow("CALIBRATE", cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    "CALIBRATE",
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

cv2.setMouseCallback("CALIBRATE", mouse_callback)

print("👉 CLICK 4 DIEM: TL → TR → BR → BL")
print("👉 BAM PHIM S DE LUU")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, FLIP_MODE)

    for i, p in enumerate(points):
        cv2.circle(frame, p, 8, (0, 0, 255), -1)
        cv2.putText(
            frame, str(i+1),
            (p[0]+10, p[1]-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    if len(points) == 4:
        cv2.polylines(
            frame,
            [np.array(points)],
            True,
            (0, 255, 0),
            3
        )

    cv2.imshow("CALIBRATE", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s') and len(points) == 4:
        np.save("roi.npy", np.array(points))
        print("✅ DA LUU roi.npy TAI:", os.getcwd())
        break

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
