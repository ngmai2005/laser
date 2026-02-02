import cv2
import numpy as np
import time
import threading
import os
from playsound import playsound
import csv
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ================= CONFIG =================
CAM_INDEX = 0
FLIP_MODE = 1

SHOT_COOLDOWN = 0.12
MAX_BULLETS = 16
TARGET_TIME = 15

# ================= SOUND =================
def play_hit_sound():
    threading.Thread(
        target=playsound,
        args=("static/sounds/hit.mp3",),
        daemon=True
    ).start()

def play_result_sound(rank):
    sound_map = {
        "GIOI": "result_gioi.mp3",
        "KHA": "result_kha.mp3",
        "DAT": "result_dat.mp3",
        "KHONG DAT": "result_khong_dat.mp3",
    }

    file = sound_map.get(rank)
    if not file:
        return

    path = os.path.join(BASE_DIR, "static/sounds", file)
    if not os.path.exists(path):
        print("❌ Missing:", path)
        return

    playsound(path)

# ================= LOAD TARGET IMAGES =================
TARGET_IMAGES_RAW = []
for i in range(1, 5):
    img = cv2.imread(f"static/images/target{i}.png")
    if img is None:
        print(f"❌ Thiếu images/target{i}.png")
        exit()
    TARGET_IMAGES_RAW.append(img)

# ================= DETECT LASER =================
def detect_laser(frame):
    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    bright = cv2.inRange(v, 240, 255)
    sat = cv2.inRange(s, 120, 255)
    base_mask = cv2.bitwise_and(bright, sat)

    red1 = cv2.inRange(hsv, (0, 120, 200), (10, 255, 255))
    red2 = cv2.inRange(hsv, (160, 120, 200), (180, 255, 255))
    green = cv2.inRange(hsv, (35, 120, 200), (90, 255, 255))
    pink = cv2.inRange(hsv, (140, 80, 200), (165, 255, 255))

    mask = base_mask & (red1 | red2 | green | pink)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, None

    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 4:
        return False, None

    (x, y), _ = cv2.minEnclosingCircle(c)
    return True, (int(x), int(y))


# ================= MAIN =================
def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cv2.namedWindow("Laser Trainer", cv2.WINDOW_NORMAL)

    is_fullscreen = False
    score = bullets = hits = 0
    phase = 1
    sub_target = 2
    round_id = 1
    MAX_ROUND = 2

    target_start = time.time()
    last_shot = 0
    laser_prev = False

    move_x, direction = 100, 1
    tts_spoken = False
    result_time = None
    result_datetime = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, FLIP_MODE)
        H, W = frame.shape[:2]

        IMAGE_SIZE = int(H * 0.55)

        TARGET_IMAGES = [
            cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            for img in TARGET_IMAGES_RAW
        ]

        laser_now, laser_pos = detect_laser(frame)
        now = time.time()

        # ================= SHOT LOGIC =================
        shot_fired = False
        if laser_now and not laser_prev and now - last_shot > SHOT_COOLDOWN and bullets < MAX_BULLETS:
            bullets += 1
            last_shot = now
            shot_fired = True

        laser_prev = laser_now
        canvas = np.zeros((H, W, 3), dtype=np.uint8)

        if laser_pos:
            lx, ly = laser_pos
            cv2.circle(canvas, (lx, ly), 6, (0, 255, 255), 2)

        # ================= PHASE 1 =================
        if phase == 1:
            tx = W // 2 - IMAGE_SIZE // 2
            ty = H // 2 - IMAGE_SIZE // 2
            canvas[ty:ty+IMAGE_SIZE, tx:tx+IMAGE_SIZE] = TARGET_IMAGES[0]

            if shot_fired and laser_pos:
                if tx < lx < tx+IMAGE_SIZE and ty < ly < ty+IMAGE_SIZE:
                    score += 1
                    hits += 1
                    play_hit_sound()
                    phase = 2
                    sub_target = 2
                    target_start = now

            elif now - target_start >= TARGET_TIME:
                phase = 2
                target_start = now

        # ================= PHASE 2 =================
        elif phase == 2:
            ty = H // 2 - IMAGE_SIZE // 2
            tx2, tx3 = 150, W - IMAGE_SIZE - 150

            if sub_target == 2:
                canvas[ty:ty+IMAGE_SIZE, tx2:tx2+IMAGE_SIZE] = TARGET_IMAGES[1]
            else:
                canvas[ty:ty+IMAGE_SIZE, tx3:tx3+IMAGE_SIZE] = TARGET_IMAGES[2]

            if shot_fired and laser_pos:
                if sub_target == 2 and tx2 < lx < tx2+IMAGE_SIZE and ty < ly < ty+IMAGE_SIZE:
                    score += 1
                    hits += 1
                    play_hit_sound()
                    sub_target = 3
                    target_start = now

                elif sub_target == 3 and tx3 < lx < tx3+IMAGE_SIZE and ty < ly < ty+IMAGE_SIZE:
                    score += 1
                    hits += 1
                    play_hit_sound()
                    phase = 4
                    target_start = now

            if sub_target == 2 and now - target_start >= 10:
                sub_target = 3
                target_start = now
            elif sub_target == 3 and now - target_start >= 5:
                phase = 4
                target_start = now

        # ================= PHASE 4 =================
        elif phase == 4:
            move_x += direction * 6
            if move_x < 50 or move_x > W - IMAGE_SIZE - 50:
                direction *= -1

            ty = H // 2 - IMAGE_SIZE // 2
            canvas[ty:ty+IMAGE_SIZE, move_x:move_x+IMAGE_SIZE] = TARGET_IMAGES[3]

            if shot_fired and laser_pos:
                if move_x < lx < move_x+IMAGE_SIZE and ty < ly < ty+IMAGE_SIZE:
                    score += 1
                    hits += 1
                    play_hit_sound()
                    if round_id < MAX_ROUND:
                        round_id += 1
                        phase = 1
                        target_start = now
                    else:
                        phase = 99
                        result_time = time.time()
                        result_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            elif now - target_start >= TARGET_TIME:
                if round_id < MAX_ROUND:
                    round_id += 1
                    phase = 1
                    target_start = now
                else:
                    phase = 99
                    result_time = time.time()
                    result_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ================= RESULT =================
        if phase == 99 or bullets >= MAX_BULLETS:
            canvas[:] = 0
            rank = "GIOI" if score >= 4 else "KHA" if score == 3 else "DAT" if score == 2 else "KHONG DAT"
            cv2.putText(canvas, f"KET QUA: {rank}", (400, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
            cv2.putText(canvas, "NHAN R DE RESET", (420, 380),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # ===== RESULT TABLE =====
            table_x, table_y = 360, 430
            row_h = 45
            col_w1, col_w2 = 260, 360

            rows = [
                ("THOI GIAN", result_datetime if result_datetime else ""),
                ("SCORE", str(score)),
                ("SO DAN", f"{bullets}/{MAX_BULLETS}"),
                ("XEP LOAI", rank),
            ]

            cv2.rectangle(
                canvas,
                (table_x, table_y),
                (table_x + col_w1 + col_w2, table_y + row_h * len(rows)),
                (40, 40, 40),
                -1
            )

            cv2.rectangle(
                canvas,
                (table_x, table_y),
                (table_x + col_w1 + col_w2, table_y + row_h * len(rows)),
                (0, 255, 255),
                2
            )

            for i, (label, value) in enumerate(rows):
                y = table_y + i * row_h

                cv2.line(
                    canvas,
                    (table_x, y),
                    (table_x + col_w1 + col_w2, y),
                    (0, 255, 255),
                    1
                )

                cv2.putText(
                    canvas,
                    label,
                    (table_x + 10, y + 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    canvas,
                    value,
                    (table_x + col_w1 + 20, y + 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 255),
                    2
                )

            if not tts_spoken and result_time and time.time() - result_time >= 0.8:
                play_result_sound(rank)
                tts_spoken = True

        cv2.putText(canvas, f"SCORE: {score}", (40, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(canvas, f"SHOT: {bullets}/{MAX_BULLETS}", (40, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow("Laser Trainer", canvas)

        if not is_fullscreen:
            cv2.setWindowProperty(
                "Laser Trainer",
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN
            )
            is_fullscreen = True

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

        if key == ord('r') and phase == 99:
            score = bullets = hits = 0
            phase = 1
            sub_target = 2
            round_id = 1
            target_start = time.time()
            last_shot = 0
            laser_prev = False
            move_x = 100
            direction = 1
            tts_spoken = False

            IMAGE_SIZE = int(H * 0.55)
            TARGET_IMAGES = [
                cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
                for img in TARGET_IMAGES_RAW
            ]

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
