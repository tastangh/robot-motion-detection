import cv2
import numpy as np

VIDEO_PATH = 'tusas-odev2-test1.mp4'
OUTPUT_TXT = 'tusas-odev2-ogr.txt'
NUM_SECONDS = 60
NUM_ROBOTS = 9
MOTION_THRESHOLD_RATIO = 0.2

# Optimize edilmiş robot bazlı eşik değerleri
ROBOT_THRESHOLDS = [
    0.0015,  # Robot 1
    0.0017,  # Robot 2
    0.0012,  # Robot 3
    0.0016,  # Robot 4
    0.0016,  # Robot 5
    0.0015,  # Robot 6
    0.0013,  # Robot 7
    0.0015,  # Robot 8
    0.0013   # Robot 9
]

# Fiziksel sıralama: Orta üst, sağ üst, sol üst, orta orta, sağ orta, sol orta, orta alt, sağ alt, sol alt
OUTPUT_ORDER = [1, 4, 7, 2, 5, 8, 0, 3, 6]

def get_grid_cells(frame):
    h, w = frame.shape[:2]
    cell_h, cell_w = h // 3, w // 3
    cells = []
    for i in range(3):
        for j in range(3):
            y1, y2 = i * cell_h, (i + 1) * cell_h
            x1, x2 = j * cell_w, (j + 1) * cell_w
            cells.append((x1, y1, x2, y2))
    return cells

def detect_fall_start_frame(cap, max_frames=150, threshold=0.01, min_global_change=6):
    print("[INFO] Düşüş başlangıcı (ani hareket) tespiti başlatılıyor...")
    ret, prev_frame = cap.read()
    if not ret:
        return 0
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    cells = get_grid_cells(prev_gray)

    for i in range(1, max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        active_cells = 0
        for (x1, y1, x2, y2) in cells:
            prev_roi = prev_gray[y1:y2, x1:x2]
            curr_roi = gray[y1:y2, x1:x2]
            diff = cv2.absdiff(prev_roi, curr_roi)
            _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
            motion = np.sum(thresh) / 255
            ratio = motion / (curr_roi.shape[0] * curr_roi.shape[1])
            if ratio > threshold:
                active_cells += 1

        if active_cells >= min_global_change:
            print(f"[INFO] Düşüş başladığı kare: {i}")
            return i

        prev_gray = gray

    print("[WARN] Düşüş başlangıcı tespit edilemedi. 0'a ayarlandı.")
    return 0

def detect_motion(prev_gray, curr_gray, epsilon):
    diff = cv2.absdiff(prev_gray, curr_gray)
    blurred = cv2.GaussianBlur(diff, (5, 5), 0)  # Gürültü azalt
    _, thresh = cv2.threshold(blurred, 25, 255, cv2.THRESH_BINARY)
    motion_pixels = np.sum(thresh) / 255
    area = prev_gray.shape[0] * prev_gray.shape[1]
    motion_ratio = motion_pixels / area
    return motion_ratio > epsilon

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Video açılamadı.")
        return

    FPS = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = NUM_SECONDS * FPS
    print(f"[INFO] FPS: {FPS}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    fall_frame = detect_fall_start_frame(cap)
    fall_frame = max(0, fall_frame - 5)
    print(f"[INFO] Düşüş başlangıcı (düzenlenmiş): {fall_frame}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, fall_frame)

    ret, frame = cap.read()
    if not ret:
        print("Başlangıç karesi okunamadı.")
        return

    cells = get_grid_cells(frame)
    prev_rois = [None] * NUM_ROBOTS
    motion_log = [[] for _ in range(NUM_ROBOTS)]
    frame_idx = 0

    while frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for idx, (x1, y1, x2, y2) in enumerate(cells):
            roi = gray[y1:y2, x1:x2]
            if prev_rois[idx] is not None:
                moved = detect_motion(prev_rois[idx], roi, epsilon=ROBOT_THRESHOLDS[idx])
                motion_log[idx].append(1 if moved else 0)
                if moved:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            prev_rois[idx] = roi

        cv2.imshow("Robot Hareket Algılama", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    print("[INFO] Hareketler .txt dosyasına yazılıyor...")
    with open(OUTPUT_TXT, 'w') as f:
        header = "Saniye\t" + "\t".join([f"Robot-{i+1}" for i in range(NUM_ROBOTS)])
        f.write(header + '\n')
        for second in range(NUM_SECONDS):
            line = f" {second+1:2})"
            for idx in OUTPUT_ORDER:
                start = second * FPS
                end = (second + 1) * FPS
                segment = motion_log[idx][start:end]
                moved = 1 if sum(segment) >= (FPS * MOTION_THRESHOLD_RATIO) else 0
                line += f"\t   {moved}"
            f.write(line + '\n')

    print(f"[✅ TAMAMLANDI] Sonuç dosyası yazıldı: {OUTPUT_TXT}")

if __name__ == '__main__':
    main()
