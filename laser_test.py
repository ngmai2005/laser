import cv2
import numpy as np
import csv
import time
import os

# ================== CẤU HÌNH ==================
CAM_INDEX = 1          # cam rời
WIDTH, HEIGHT = 1920, 1080   # độ phân giải cao nhất -> góc rộng nhất
FLIP_MODE = 1          # 1 = lật ngang (đa số cam rời)
FPS_LIMIT = 120

TARGET_WIDTH = 800
TARGET_HEIGHT = 800

CALIB_FILE = "calibration_points.npy"
REPORT_FILE = "report.csv"

# ================== HÀM HỖ TRỢ ==================
def load_or_calibrate(cap):
    """
    Nếu chưa có calibration_points.npy thì cho click 4 điểm
    """
    if os.path.exists(CALIB_FILE):
        try:
            pts = np.load(CALIB_FILE).astype(np.float32)
            if pts.shape == (4, 2):
                print("✅ Load calibration thành công")
                return pts
        except:
            pass

    print("⚠ Chưa có calibration, click 4 điểm theo thứ tự:")
    print("TOP-LEFT → TOP-RIGHT → BOTTOM-RIGHT → BOTTOM-LEFT")

    points = []

    def mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([x, y])
            print(f"Point {len(points)}: {x}, {y}")

    cv2.namedWindow("CALIBRATION")
    cv2.setMouseCallback("CALIBRATION", mouse)

    while True:
        cap.grab()
        ret, frame = cap.retrieve()
        if not ret:
            continue

        frame = cv2.flip(frame, FLIP_MODE)

        for p in points:
            cv2.circle(frame, tuple(p), 6, (0, 255, 0), -1)

        cv2.imshow("CALIBRATION", frame)

        if len(points) == 4:
            pts = np.array(points, dtype=np.float32)
            np.save(CALIB_FILE, pts)
            print("✅ Đã lưu calibration_points.npy")
            break

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyWindow("CALIBRATION")
    return pts


def get_perspective_matrix(src_pts):
    dst_pts = np.array([
        [0, 0],
        [TARGET_WIDTH, 0],
        [TARGET_WIDTH, TARGET_HEIGHT],
        [0, TARGET_HEIGHT]
    ], dtype=np.float32)

    return cv2.getPerspectiveTransform(src_pts, dst_pts)


def detect_laser(frame):
    """
    Detect laser đỏ / hồng
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # đỏ + hồng
    lower1 = np.array([0, 120, 200])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([160, 120, 200])
    upper2 = np.array([179, 255, 255])

    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not cnts:
        return None

    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 5:
        return None

    M = cv2.moments(c)
    if M["m00"] == 0:
        return None

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return cx, cy


def score_point(x, y):
    """
    Chia bia đơn giản: tâm càng gần thì điểm càng cao
    """
    cx, cy = TARGET_WIDTH // 2, TARGET_HEIGHT // 2
    dist = np.hypot(x - cx, y - cy)

    if dist < 50:
        return 10, "CENTER"
    elif dist < 150:
        return 8, "INNER"
    elif dist < 300:
        return 5, "OUTER"
    else:
        return 0, "MISS"


# ================== MAIN ==================
def main():
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS_LIMIT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("❌ Không mở được cam")
        return

    src_pts = load_or_calibrate(cap)
    M = get_perspective_matrix(src_pts)

    # tạo report.csv
    if not os.path.exists(REPORT_FILE):
        with open(REPORT_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "x", "y", "region", "score", "target"
            ])

    last_shot_time = 0

    while True:
        cap.grab()
        ret, frame = cap.retrieve()
        if not ret:
            continue

        frame = cv2.flip(frame, FLIP_MODE)
        warped = cv2.warpPerspective(frame, M, (TARGET_WIDTH, TARGET_HEIGHT))

        laser = detect_laser(warped)

        if laser:
            x, y = laser
            score, region = score_point(x, y)

            now = time.time()
            if now - last_shot_time > 0.25:  # chống ghi trùng
                last_shot_time = now

                with open(REPORT_FILE, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        time.strftime("%Y-%m-%d %H:%M:%S"),
                        x, y, region, score, "TARGET_1"
                    ])

            cv2.circle(warped, (x, y), 8, (0, 0, 255), -1)
            cv2.putText(warped, f"{score}", (x+10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow("LASER TARGET", warped)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
