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
LASER_MIN_AREA = 1
TEMPORAL_FRAMES = 3
WIDTH, HEIGHT = 1280, 720
LASER_HOLD_MS = 80

# ================= CAMERA =================
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)

cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
cap.set(cv2.CAP_PROP_EXPOSURE, -6)
cap.set(cv2.CAP_PROP_GAIN, 0)


latest_frame = None
frame_lock = threading.Lock()
running = True

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
    global laser_pos, last_laser_time

    while running:
        with frame_lock:
            if latest_frame is None:
                time.sleep(0.005)
                continue
            frame = latest_frame.copy()

        blur = cv2.GaussianBlur(frame, (3, 3), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Hue đỏ – nới rộng cho camera lệch màu
        red1 = cv2.inRange(hsv, (0, 80, 50), (15, 255, 255))
        red2 = cv2.inRange(hsv, (165, 80, 50), (180, 255, 255))
        mask = cv2.bitwise_or(red1, red2)

        # Chỉ giữ pixel đủ bão hòa (laser rất bão hòa)
        sat_mask = cv2.inRange(s, 90, 255)
        mask = cv2.bitwise_and(mask, sat_mask)

        # mask = cv2.bitwise_and(bright, sat)
        # mask = cv2.bitwise_and(mask, red1 | red2)

        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.imshow("laser mask", mask)
        cv2.waitKey(1)

        pos = None
        max_brightness = 0

        for c in cnts:
            area = cv2.contourArea(c)
            if 3 < area < 80:  
                x, y, w, h = cv2.boundingRect(c)
                brightness = v[y:y+h, x:x+w].mean()

                if brightness > max_brightness:
                    max_brightness = brightness
                    pos = (int(x + w//2), int(y + h//2))

        now = time.time() * 1000

        if pos:
            laser_buffer.append(pos)
            if len(laser_buffer) > TEMPORAL_FRAMES:
                laser_buffer.pop(0)

            avg_x = int(sum(p[0] for p in laser_buffer) / len(laser_buffer))
            avg_y = int(sum(p[1] for p in laser_buffer) / len(laser_buffer))

            laser_pos = [avg_x, avg_y]
            last_laser_time = now

        else:
            if now - last_laser_time > LASER_HOLD_MS:
                laser_pos = None
                laser_buffer.clear()

        time.sleep(0.01)  # ~100Hz
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

# ================= RUN =================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
