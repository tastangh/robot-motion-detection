import cv2
import numpy as np
import os

# Girdi ve çıktı yolları (gerekiyorsa değiştirebilirsin)
VIDEO_PATH = "odev2-videolar/tusas-odev2-test1.mp4"
OUTPUT_PATH = "odev2-videolar/tusas-odev2-ogr1.txt"  # kontrol scripti bunu bekliyor
SECONDS = 60
NUM_ROBOTS = 9
GRID_ROWS, GRID_COLS = 3, 3
FPS_FALLBACK = 30

def detect_fall_start(cap):
    """Robotların yere düştüğü anı titreşimden tespit et."""
    print("[INFO] Başlangıç zamanı tespit ediliyor...")
    history = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if len(history) > 0:
            diff = cv2.absdiff(history[-1], gray)
            score = np.sum(diff) / 255
            if score > 500000:
                print("[INFO] Robot düşüşü tespit edildi.")
                return cap.get(cv2.CAP_PROP_POS_FRAMES)
        history.append(gray)
    return 0

def get_robot_cells(frame):
    """Görüntüyü 3x3 hücreye bölerek robot bölgelerini döndür."""
    h, w = frame.shape[:2]
    cell_w, cell_h = w // GRID_COLS, h // GRID_ROWS
    boxes = []
    for i in range(GRID_ROWS):
        for j in range(GRID_COLS):
            x = j * cell_w
            y = i * cell_h
            boxes.append((x, y, cell_w, cell_h))
    return boxes

def compute_motion_score(prev, curr):
    """İki kare arasında hareket miktarını hesapla."""
    diff = cv2.absdiff(prev, curr)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    score = np.sum(thresh) / 255
    return score

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[ERROR] Video açılamadı: {VIDEO_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or FPS_FALLBACK
    total_frames = int(SECONDS * fps)

    fall_start_frame = int(detect_fall_start(cap))
    cap.set(cv2.CAP_PROP_POS_FRAMES, fall_start_frame)

    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Başlangıç karesi okunamadı.")
        return

    robot_boxes = get_robot_cells(frame)
    prev_rois = [None] * NUM_ROBOTS
    motion_buffer = [[0] * SECONDS for _ in range(NUM_ROBOTS)]

    frame_index = 0
    while frame_index < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        second = int(frame_index / fps)
        for idx, (x, y, w, h) in enumerate(robot_boxes):
            roi = frame[y:y+h, x:x+w]
            if prev_rois[idx] is not None:
                score = compute_motion_score(prev_rois[idx], roi)
                if score > 800:
                    motion_buffer[idx][second] += 1
                if score > 800:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            prev_rois[idx] = roi

        frame_index += 1

    # %20 kuralına göre hareketli saniyeler 1, değilse 0
    result = []
    for robot_id in range(NUM_ROBOTS):
        line = []
        for sec in range(SECONDS):
            if motion_buffer[robot_id][sec] >= int(fps * 0.2):
                line.append("1")
            else:
                line.append("0")
        result.append(line)

    # Kontrol scriptine uygun formatta txt dosyası yaz
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        f.write("Saniye\t" + "\t".join([f"Robot-{i+1}" for i in range(NUM_ROBOTS)]) + "\n")
        for sec in range(SECONDS):
            line = [result[robot_id][sec] for robot_id in range(NUM_ROBOTS)]
            f.write(f"{sec+1:3d})\t" + "\t".join(line) + "\n")

    print(f"[INFO] Sonuç dosyası yazıldı: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
