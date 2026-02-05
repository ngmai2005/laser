import cv2, numpy as np, json, atexit, threading, time
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# ================= APP =================
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# ================= CONFIG =================
CAM_INDEX = 0
FLIP_MODE = 1
LASER_MIN_AREA = 3
LASER_MAX_AREA = 160
TEMPORAL_FRAMES = 4
WIDTH, HEIGHT = 1280, 720
LASER_HOLD_MS = 100

# ================= CAMERA =================
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)

cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
cap.set(cv2.CAP_PROP_EXPOSURE, -7)
cap.set(cv2.CAP_PROP_GAIN, 0)


latest_frame = None
frame_lock = threading.Lock()
running = True
latest_mask = None

# ================= DATA =================
laser_buffer = []
laser_pos = None
last_laser_time = 0
calibration = {
    "top_left": [0, 0],
    "top_right": [WIDTH, 0],
    "bottom_left": [0, HEIGHT],
    "bottom_right": [WIDTH, HEIGHT]
}

# ================= CAMERA LOOP =================
def camera_loop():
    global latest_frame
    while running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        frame = cv2.flip(frame, FLIP_MODE)
        with frame_lock:
            latest_frame = frame.copy()

threading.Thread(target=camera_loop, daemon=True).start()

# ================= LASER LOOP =================
def laser_loop():
    global laser_pos, last_laser_time, laser_buffer, latest_mask

    kernel = np.ones((2, 2), np.uint8)

    while running:
        with frame_lock:
            if latest_frame is None:
                time.sleep(0.005)
                continue
            frame = latest_frame.copy()

        blur = cv2.GaussianBlur(frame, (3, 3), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        red1 = cv2.inRange(hsv, (0, 60, 160), (10, 255, 255))
        red2 = cv2.inRange(hsv, (170, 60, 160), (180, 255, 255))
        red_mask = cv2.bitwise_or(red1, red2)

        bright_mask = cv2.inRange(v, 200, 255)
        sat_mask = cv2.inRange(s, 80, 255)

        mask = cv2.bitwise_and(red_mask, bright_mask)
        mask = cv2.bitwise_and(mask, sat_mask)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)

        latest_mask = mask.copy()   # 👈 QUAN TRỌNG

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        pos = None
        best_score = 0

        for c in cnts:
            area = cv2.contourArea(c)
            if not (LASER_MIN_AREA <= area <= LASER_MAX_AREA):
                continue

            x, y, w, h = cv2.boundingRect(c)
            brightness = v[y:y+h, x:x+w].mean()
            score = brightness / area

            if score > best_score:
                best_score = score
                pos = (x + w // 2, y + h // 2)

        now = time.time() * 1000

        if pos:
            laser_buffer.append(pos)
            if len(laser_buffer) > TEMPORAL_FRAMES:
                laser_buffer.pop(0)

            laser_pos = [
                int(sum(p[0] for p in laser_buffer) / len(laser_buffer)),
                int(sum(p[1] for p in laser_buffer) / len(laser_buffer))
            ]
            last_laser_time = now
        else:
            if now - last_laser_time > LASER_HOLD_MS:
                laser_pos = None
                laser_buffer.clear()

        time.sleep(0.01)

    if pos:
        print("🔴 laser:", pos, "area:", cv2.contourArea(c))
    else:
        print("⚫ no laser")

threading.Thread(target=laser_loop, daemon=True).start()

# ================= VIDEO STREAM =================
def gen_frames():
    while True:
        with frame_lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()
        ret, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               buffer.tobytes() + b"\r\n")

@app.get("/video")
def video():
    return StreamingResponse(
        gen_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# ================= LASER DETECT =================
@app.get("/laser")
def get_laser():
    # global laser_pos, last_laser_time

    # with frame_lock:
    #     if latest_frame is None:
    #         return {"x": None, "y": None}
    #     frame = latest_frame.copy()

    # blur = cv2.GaussianBlur(frame, (5, 5), 0)
    # hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    # h, s, v = cv2.split(hsv)

    # bright = cv2.inRange(v, 160, 255)
    # sat= cv2.inRange(s, 60, 255)
    # red1 = cv2.inRange(hsv, (0, 120, 200), (10, 255, 255))
    # red2 = cv2.inRange(hsv, (160, 120, 200), (180, 255, 255))
    # # green = cv2.inRange(hsv, (35, 120, 200), (90, 255, 255))

    # mask = cv2.bitwise_and(bright, sat)
    # mask = cv2.bitwise_and(mask, red1 | red2)
    # cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # pos = None
    # if cnts:
    #     c = max(cnts, key=cv2.contourArea)
    #     if cv2.contourArea(c) >= LASER_MIN_AREA:
    #         (x, y), _ = cv2.minEnclosingCircle(c)
    #         pos = (int(x), int(y))

    # now = time.time() * 1000

    # if pos:
    #     laser_pos = [pos[0], pos[1]]
    #     last_laser_time = now
    # else:
    #     if now - last_laser_time > 120:
    #         laser_pos = None

    return {
        "x": laser_pos[0] if laser_pos else None,
        "y": laser_pos[1] if laser_pos else None
    }

# ================= CALIBRATE =================
@app.get("/calibrate")
def calibrate():
    global calibration
    calibration = {
        "top_left": [0, 0],
        "top_right": [WIDTH, 0],
        "bottom_left": [0, HEIGHT],
        "bottom_right": [WIDTH, HEIGHT]
    }
    with open("calibration.json", "w") as f:
        json.dump(calibration, f)
    return {"status": "ok"}

# ================= INDEX =================
@app.get("/")
def index():
    return FileResponse("index.html")

# ================= CLEANUP =================
def release_camera():
    global running
    running = False
    cap.release()

atexit.register(release_camera)

def debug_window():
    while running:
        if latest_mask is not None:
            cv2.imshow("LASER MASK", latest_mask)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyAllWindows()

# ================= RUN =================
if __name__ == "__main__":
    threading.Thread(target=laser_loop, daemon=True).start()
    debug_window()
