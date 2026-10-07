"""
LOVE HAND — versi smooth + zodiak + mengikuti gerakan tangan
=======================================================================
Kanvas partikel memenuhi seluruh jendela. Kamera tampil sebagai kotak
kecil landscape (4:3) di pojok kiri atas, seperti pada video referensi.

Cara pakai:
    1. pip install opencv-python mediapipe numpy
    2. Ganti PARTNER_NAME dan BIRTHDAY_MESSAGE di bawah.
    3. Jalankan:  python particle_love.py
    4. Tekan 'q' untuk keluar.

Gesture (SATU tangan, hadapkan telapak ke kamera):
    0 jari (kepal / fist)   -> partikel membentuk HATI
    1 jari (telunjuk)       -> partikel membentuk simbol zodiak LIBRA (8 Okt)
    2 jari (peace / victory)-> partikel membentuk "I ALWAYS LOVE YOU <NAMA>"
    3 jari                  -> partikel membentuk ucapan ulang tahun
    4 / 5 jari / tanpa tangan -> partikel menyebar bebas

BARU: saat bentuk sedang terbentuk (heart / zodiac / love / bday),
      seluruh bentuk akan bergeser mengikuti posisi telapak tangan
      di dalam frame kamera, dengan smoothing supaya tetap halus.
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import time

# ============================================================
# KONFIGURASI — GANTI DI SINI
# ============================================================
PARTNER_NAME = "SHEILA"
BIRTHDAY_MESSAGE = "HAPPY BIRTHDAY SAYANG"

LOVE_TEXT = f"I ALWAYS LOVE YOU {PARTNER_NAME}"

CANVAS_W, CANVAS_H = 900, 600      # ukuran seluruh jendela (kanvas partikel)
CAM_W, CAM_H = 260, 195            # kotak kamera kecil, rasio 4:3 seperti video referensi
CAM_MARGIN = 18                    # jarak kotak kamera dari tepi kiri-atas

FONT = cv2.FONT_HERSHEY_DUPLEX
N_PARTICLES = 2400

CAPTURE_W, CAPTURE_H = 640, 480    # resolusi tangkapan asli, dipakai untuk deteksi tangan

MODE_DEBOUNCE_FRAMES = 5           # gesture harus konsisten sekian frame sebelum berpindah

# --- pengaturan "bentuk mengikuti tangan" ---
FOLLOW_HAND = True                 # matikan (False) untuk kembali ke perilaku lama (diam di tengah)
FOLLOW_EASE_RATE = 6.0             # makin besar -> makin cepat/tegang mengejar tangan
FOLLOW_RETURN_EASE_RATE = 3.0      # kecepatan kembali ke tengah saat tangan hilang
FOLLOW_MAX_OFFSET_X = CANVAS_W * 0.32   # batas geser maksimum horizontal (supaya bentuk tak keluar layar)
FOLLOW_MAX_OFFSET_Y = CANVAS_H * 0.30   # batas geser maksimum vertikal

# ============================================================
# MEDIAPIPE HANDS
# ============================================================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=0,             # model ringan -> lebih cepat & lebih smooth
    min_detection_confidence=0.65,
    min_tracking_confidence=0.55,
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_H)
cap.set(cv2.CAP_PROP_FPS, 30)


# ============================================================
# TARGET SHAPES -> ARRAYS OF POINTS (dihitung sekali di awal)
# ============================================================
def fit_font_scale(text, max_width, thickness=3):
    scale = 3.5
    while scale > 0.3:
        (tw, _), _ = cv2.getTextSize(text, FONT, scale, thickness)
        if tw <= max_width:
            return scale
        scale -= 0.05
    return 0.3


def text_to_points(text, n_points):
    thickness = 3
    scale = fit_font_scale(text, int(CANVAS_W * 0.85), thickness)
    (tw, th), baseline = cv2.getTextSize(text, FONT, scale, thickness)

    mask = np.zeros((CANVAS_H, CANVAS_W), dtype=np.uint8)
    x = (CANVAS_W - tw) // 2
    y = (CANVAS_H + th) // 2
    cv2.putText(mask, text, (x, y), FONT, scale, 255, thickness, cv2.LINE_AA)

    ys, xs = np.nonzero(mask)
    pts = np.column_stack((xs, ys)).astype(np.float32)
    if len(pts) == 0:
        pts = np.array([[CANVAS_W / 2, CANVAS_H / 2]], dtype=np.float32)

    idx = np.random.choice(len(pts), size=n_points, replace=len(pts) < n_points)
    return pts[idx]


def heart_points(n_points, center, scale):
    t = np.random.uniform(0, 2 * np.pi, n_points)
    r = np.sqrt(np.random.uniform(0, 1, n_points))

    x = 16 * np.sin(t) ** 3
    y = -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))

    x *= r
    y *= r
    x = x * scale + center[0]
    y = y * scale + center[1]
    return np.column_stack((x, y)).astype(np.float32)


def libra_points(n_points, center, scale):
    """Simbol zodiak Libra (garis + lengkung, seperti timbangan) -- untuk 8 Oktober."""
    pad = 200
    mask = np.zeros((pad * 2, pad * 2), dtype=np.uint8)
    cx, cy = pad, pad
    bar_half = 70
    thickness = 12
    cv2.line(mask, (cx - bar_half, cy - 45), (cx + bar_half, cy - 45), 255, thickness, cv2.LINE_AA)
    cv2.line(mask, (cx, cy - 45), (cx, cy - 8), 255, thickness, cv2.LINE_AA)
    cv2.ellipse(mask, (cx, cy + 8), (bar_half, 45), 0, 0, 180, 255, thickness, cv2.LINE_AA)
    cv2.line(mask, (cx - bar_half, cy + 53), (cx + bar_half, cy + 53), 255, thickness, cv2.LINE_AA)

    ys, xs = np.nonzero(mask)
    pts = np.column_stack((xs, ys)).astype(np.float32)
    pts -= [cx, cy]
    pts *= scale
    pts += center

    idx = np.random.choice(len(pts), size=n_points, replace=len(pts) < n_points)
    return pts[idx].astype(np.float32)


CENTER = (CANVAS_W / 2, CANVAS_H / 2)
CENTER_ARR = np.array(CENTER, dtype=np.float32)

target_love = text_to_points(LOVE_TEXT, N_PARTICLES)
target_bday = text_to_points(BIRTHDAY_MESSAGE, N_PARTICLES)
target_heart = heart_points(N_PARTICLES, (CENTER[0], CENTER[1] - 40), scale=min(CANVAS_W, CANVAS_H) / 40)
target_zodiac = libra_points(N_PARTICLES, CENTER, scale=1.9)

MODE_TARGETS = {
    "heart": target_heart,
    "zodiac": target_zodiac,
    "love": target_love,
    "bday": target_bday,
}

MODE_COLORS = {
    "scatter": np.array([235, 225, 215], dtype=np.float32),  # icy white-blue
    "heart":   np.array([150, 110, 255], dtype=np.float32),  # pink / rose
    "zodiac":  np.array([200, 220, 255], dtype=np.float32),  # soft champagne gold
    "love":    np.array([235, 210, 70], dtype=np.float32),   # cyan
    "bday":    np.array([90, 210, 255], dtype=np.float32),   # gold
}

MODE_LABELS = {
    "scatter": "Tangan terbuka / tanpa tangan -- partikel menyebar",
    "heart":   "0 jari (kepal) -- membentuk HATI (ikuti tangan)",
    "zodiac":  "1 jari -- simbol zodiak LIBRA (ikuti tangan)",
    "love":    "2 jari (peace) -- membentuk pesan cinta (ikuti tangan)",
    "bday":    "3 jari -- membentuk ucapan ulang tahun (ikuti tangan)",
}

# landmark tangan yang dipakai sebagai "pusat telapak" untuk pelacakan
# 0 = pergelangan, 5/9/13/17 = pangkal jari telunjuk/tengah/manis/kelingking
PALM_LANDMARK_IDS = [0, 5, 9, 13, 17]

# ============================================================
# PARTICLE STATE
# ============================================================
rng = np.random.default_rng()

scatter_pos = rng.normal(
    loc=[CENTER[0], CENTER[1]], scale=[CANVAS_W * 0.22, CANVAS_H * 0.22],
    size=(N_PARTICLES, 2)
).astype(np.float32)
np.clip(scatter_pos[:, 0], 0, CANVAS_W, out=scatter_pos[:, 0])
np.clip(scatter_pos[:, 1], 0, CANVAS_H, out=scatter_pos[:, 1])

scatter_vel = rng.uniform(-20, 20, size=(N_PARTICLES, 2)).astype(np.float32)  # px/detik
pos = scatter_pos.copy()

twinkle_phase = rng.uniform(0, math.pi * 2, size=N_PARTICLES).astype(np.float32)
twinkle_speed = rng.uniform(1.0, 3.0, size=N_PARTICLES).astype(np.float32)
current_color = MODE_COLORS["scatter"].copy()

# posisi pusat bentuk yang dilacak (di-smoothing), mulai dari tengah kanvas
tracked_center = CENTER_ARR.copy()


def landmark_dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def count_fingers(landmarks):
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    count = 0
    for tip, pip in zip(tips, pips):
        if landmarks[tip].y < landmarks[pip].y - 0.02:
            count += 1
    if landmark_dist(landmarks[4], landmarks[17]) > landmark_dist(landmarks[3], landmarks[17]) + 0.02:
        count += 1
    return count


def palm_center_normalized(landmarks):
    """Rata-rata beberapa titik pangkal telapak -> koordinat 0..1 yang stabil."""
    xs = [landmarks[i].x for i in PALM_LANDMARK_IDS]
    ys = [landmarks[i].y for i in PALM_LANDMARK_IDS]
    return sum(xs) / len(xs), sum(ys) / len(ys)


# ============================================================
# MAIN LOOP
# ============================================================
start_time = time.time()
prev_time = start_time

active_mode = "scatter"
pending_mode = "scatter"
pending_count = 0

activated = False           # partikel belum tampil sampai tangan dibuka lebar
ACTIVATE_FINGER_COUNT = 5   # harus 5 jari (telapak terbuka lebar) untuk memulai

print("Tekan 'q' pada jendela video untuk keluar.")
print("Buka telapak tangan lebar-lebar (5 jari) untuk memulai partikel...")

while True:
    ok, frame = cap.read()
    if not ok:
        print("Tidak bisa membaca kamera.")
        break

    now = time.time()
    dt = max(1e-3, min(0.05, now - prev_time))  # clamp supaya tidak meloncat saat lag
    prev_time = now

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (CAPTURE_W, CAPTURE_H))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    detected_mode = "scatter"
    raw_finger_count = None
    hand_norm_pos = None  # (x, y) 0..1 posisi telapak di frame, jika tangan terdeteksi
    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0].landmark
        n_fingers = count_fingers(landmarks)
        raw_finger_count = n_fingers
        detected_mode = {0: "heart", 1: "zodiac", 2: "love", 3: "bday"}.get(n_fingers, "scatter")
        hand_norm_pos = palm_center_normalized(landmarks)

        mp_drawing.draw_landmarks(
            frame,
            results.multi_hand_landmarks[0],
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )

    # ---- gerbang aktivasi: partikel baru muncul setelah tangan dibuka lebar ----
    if not activated:
        if raw_finger_count == ACTIVATE_FINGER_COUNT:
            activated = True
            pending_mode = "scatter"
            pending_count = MODE_DEBOUNCE_FRAMES  # langsung stabil begitu aktif

    # ---- debounce: gesture harus stabil beberapa frame sebelum mode berpindah ----
    if detected_mode == pending_mode:
        pending_count += 1
    else:
        pending_mode = detected_mode
        pending_count = 1

    if activated and pending_count >= MODE_DEBOUNCE_FRAMES:
        active_mode = pending_mode

    cam_small = cv2.resize(frame, (CAM_W, CAM_H))

    # ---- update wandering scatter positions (frame-rate independent) ----
    scatter_vel += rng.uniform(-8, 8, size=scatter_vel.shape).astype(np.float32) * dt
    speed = np.linalg.norm(scatter_vel, axis=1, keepdims=True) + 1e-6
    max_speed = 55.0
    scatter_vel = scatter_vel / speed * np.minimum(speed, max_speed)
    scatter_pos += scatter_vel * dt

    # margin di dalam kanvas tempat awan partikel boleh menyebar bebas
    margin = 40
    lo_x, hi_x = margin, CANVAS_W - margin
    lo_y, hi_y = margin, CANVAS_H - margin

    # pantulkan kecepatan saat menyentuh batas, JANGAN ditarik ke tengah terus-menerus
    hit_lo_x = scatter_pos[:, 0] < lo_x
    hit_hi_x = scatter_pos[:, 0] > hi_x
    hit_lo_y = scatter_pos[:, 1] < lo_y
    hit_hi_y = scatter_pos[:, 1] > hi_y

    scatter_vel[hit_lo_x | hit_hi_x, 0] *= -1
    scatter_vel[hit_lo_y | hit_hi_y, 1] *= -1

    scatter_pos[:, 0] = np.clip(scatter_pos[:, 0], lo_x, hi_x)
    scatter_pos[:, 1] = np.clip(scatter_pos[:, 1], lo_y, hi_y)

    # ---- lacak posisi telapak tangan -> geser pusat bentuk (heart/zodiac/love/bday) ----
    is_shape_mode = active_mode != "scatter"
    if FOLLOW_HAND and is_shape_mode and hand_norm_pos is not None:
        raw_center = np.array(
            [hand_norm_pos[0] * CANVAS_W, hand_norm_pos[1] * CANVAS_H],
            dtype=np.float32,
        )
        ease = 1.0 - math.exp(-FOLLOW_EASE_RATE * dt)
    else:
        # tangan hilang atau sedang mode scatter -> kembali pelan-pelan ke tengah
        raw_center = CENTER_ARR
        ease = 1.0 - math.exp(-FOLLOW_RETURN_EASE_RATE * dt)

    tracked_center += (raw_center - tracked_center) * ease

    offset = tracked_center - CENTER_ARR
    offset[0] = np.clip(offset[0], -FOLLOW_MAX_OFFSET_X, FOLLOW_MAX_OFFSET_X)
    offset[1] = np.clip(offset[1], -FOLLOW_MAX_OFFSET_Y, FOLLOW_MAX_OFFSET_Y)

    if is_shape_mode:
        desired_target = MODE_TARGETS[active_mode] + offset
    else:
        desired_target = scatter_pos

    # smoothing berbasis waktu -> mulus di frame rate berapa pun
    rate = 5.0 if active_mode == "scatter" else 6.5
    ease_pos = 1.0 - math.exp(-rate * dt)
    pos += (desired_target - pos) * ease_pos

    jitter_amp = 1.0
    pos_render = pos + rng.uniform(-jitter_amp, jitter_amp, size=pos.shape).astype(np.float32)

    target_color = MODE_COLORS[active_mode]
    color_ease = 1.0 - math.exp(-4.0 * dt)
    current_color += (target_color - current_color) * color_ease

    # ---- render particle canvas ----
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

    if activated:
        t = now - start_time
        twinkle = 0.65 + 0.35 * np.sin(twinkle_phase + t * twinkle_speed)
        colors = np.clip(current_color[None, :] * twinkle[:, None], 0, 255).astype(np.uint8)

        xs = np.clip(pos_render[:, 0].astype(np.int32), 0, CANVAS_W - 1)
        ys = np.clip(pos_render[:, 1].astype(np.int32), 0, CANVAS_H - 1)
        canvas[ys, xs] = colors

        kernel = np.ones((3, 3), np.uint8)
        canvas = cv2.dilate(canvas, kernel, iterations=1)
    glow = cv2.GaussianBlur(canvas, (0, 0), sigmaX=5)
    canvas = cv2.add(canvas, (glow.astype(np.float32) * 0.6).astype(np.uint8))

    if activated:
        label = MODE_LABELS[active_mode]
    else:
        label = "Buka telapak tangan lebar-lebar (5 jari) untuk memulai..."

    cv2.putText(canvas, label, (16, CANVAS_H - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (190, 190, 190), 1, cv2.LINE_AA)

    # ---- kotak kamera kecil landscape di pojok kiri atas ----
    x0, y0 = CAM_MARGIN, CAM_MARGIN
    x1, y1 = x0 + CAM_W, y0 + CAM_H
    canvas[y0:y1, x0:x1] = cam_small
    cv2.rectangle(canvas, (x0 - 2, y0 - 2), (x1 + 2, y1 + 2), (90, 90, 90), 2)

    cv2.imshow("LOVE HAND", canvas)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()