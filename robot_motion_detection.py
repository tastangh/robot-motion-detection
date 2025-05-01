import cv2
import numpy as np
import os

# === Ayarlar ===
VIDEO_PATH = "odev2-videolar/tusas-odev2-test1.mp4"
OUTPUT_TXT = "odev2-videolar/tusas-odev2-ogr1.txt"
SECONDS = 60
GRID_ROWS, GRID_COLS = 3, 3
HARRIS_DIFF_THRESHOLD = 30
MOTION_RATIO = 0.2  # %20

# Kamera kalibrasyon (örnek)
camera_matrix = np.array([[800, 0, 320],
                          [0, 800, 240],
                          [0,   0,   1]], dtype=np.float32)
dist_coeffs = np.array([-0.25, 0.1, 0.0, 0.0, 0.0], dtype=np.float32)

# TXT dosyasına yazılırken kullanılacak robot sıralaması
# (Orta üst, orta orta, orta alt, sağ üst, sağ orta, sağ alt, sol üst, sol orta, sol alt)
txt_order = [1, 4, 7, 2, 5, 8, 0, 3, 6]

def undistort_frame(frame, K, dist):
    h, w = frame.shape[:2]
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), 1, (w,h))
    dst = cv2.undistort(frame, K, dist, None, new_K)
    x, y, w, h = roi
    return dst[y:y+h, x:x+w]

def detect_drop_frame(cap):
    history = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = undistort_frame(frame, camera_matrix, dist_coeffs)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)
        if len(history) > 0:
            diff = cv2.absdiff(history[-1], gray)
            score = np.sum(diff) / 255
            if score > 200000:
                stable_counter = 0
                while stable_counter < 15:
                    ret2, f2 = cap.read()
                    if not ret2:
                        break
                    f2 = undistort_frame(f2, camera_matrix, dist_coeffs)
                    g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
                    g2 = cv2.GaussianBlur(g2, (5,5), 0)
                    d2 = cv2.absdiff(gray, g2)
                    s2 = np.sum(d2) / 255
                    if s2 < 100000:
                        stable_counter += 1
                    else:
                        break
                if stable_counter >= 15:
                    return frame_idx
        history.append(gray)
        frame_idx += 1
    return 0

def split_cells(frame):
    h, w = frame.shape[:2]
    cell_h = h // GRID_ROWS
    cell_w = w // GRID_COLS
    cells = []
    for i in range(GRID_ROWS):
        for j in range(GRID_COLS):
            x1, y1 = j * cell_w, i * cell_h
            x2, y2 = x1 + cell_w, y1 + cell_h
            cells.append((x1, y1, x2, y2))
    return cells

def get_robot_regions(cells):
    robot_regions = []
    for (x1, y1, x2, y2) in cells:
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cell_w = x2 - x1
        scale = cell_w / 1.2  # px/metre
        robot_half = int((0.707 * scale) / 2)
        rx1, ry1 = cx - robot_half, cy - robot_half
        rx2, ry2 = cx + robot_half, cy + robot_half
        robot_regions.append((rx1, ry1, rx2, ry2))
    return robot_regions

def harris_score(gray):
    blurred = cv2.GaussianBlur(gray, (5,5), 1)
    dst = cv2.cornerHarris(np.float32(blurred), 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    corners = dst > 0.01 * dst.max()
    return np.sum(corners)

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    drop_start = detect_drop_frame(cap)
    print(f"[INFO] Robot yere düşme sonrası başlama frame: {drop_start}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, drop_start)
    ret, first_frame = cap.read()
    if not ret:
        print("Video okunamadı.")
        return
    first_frame = undistort_frame(first_frame, camera_matrix, dist_coeffs)
    h, w = first_frame.shape[:2]

    cells = split_cells(first_frame)
    robot_regions = get_robot_regions(cells)

    motion_matrix = np.zeros((SECONDS, 9), dtype=int)
    frame_buffer = [[] for _ in range(9)]
    prev_scores = [None for _ in range(9)]
    frame_idx = 0

    while frame_idx < int(SECONDS * fps):
        ret, frame = cap.read()
        if not ret:
            break

        frame = undistort_frame(frame, camera_matrix, dist_coeffs)
        second = int(frame_idx / fps)

        for i, (rx1, ry1, rx2, ry2) in enumerate(robot_regions):
            roi = frame[ry1:ry2, rx1:rx2]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            score = harris_score(gray)
            if prev_scores[i] is not None:
                diff = abs(score - prev_scores[i])
                if diff > HARRIS_DIFF_THRESHOLD:
                    frame_buffer[i].append(1)
                else:
                    frame_buffer[i].append(0)
            prev_scores[i] = score

        if frame_idx % int(fps) == 0 and frame_idx > 0:
            sec = int(frame_idx / fps) - 1
            for i in range(9):
                ratio = sum(frame_buffer[i]) / len(frame_buffer[i])
                if ratio >= MOTION_RATIO:
                    motion_matrix[sec][i] = 1
            frame_buffer = [[] for _ in range(9)]

        for i, (rx1, ry1, rx2, ry2) in enumerate(robot_regions):
            if len(frame_buffer[i]) > 0 and frame_buffer[i][-1] == 1:
                cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0,255,0), 2)

        cv2.imshow("Robot Hareket Takibi", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    # TXT çıktısı (istenen sırada ve formatta)
    with open(OUTPUT_TXT, "w") as f:
        # Başlık (hedef formatla aynı)
        f.write("Saniye\tRobot-1 Robot-2 Robot-3 Robot-4 Robot-5 Robot-6 Robot-7 Robot-8 Robot-9\n")

        for i in range(SECONDS):
            # 1. Her bir değeri 4 karakter genişliğinde sağa hizalı string olarak formatla
            formatted_values = [f"{motion_matrix[i][j]:>4}" for j in range(9)]

            # 2. Bu formatlanmış stringleri aralarına "\t" (tab) koyarak birleştir
            line_data = "\t".join(formatted_values)

            # 3. Satır numarasını formatla, ilk tabı ekle ve birleştirilmiş veriyi ekle
            f.write(f"{i+1:3})\t{line_data}\n") # Satır numarasından sonra zaten bir tab var

        print(f"[INFO] TXT çıktısı tamamlandı: {OUTPUT_TXT}")

if __name__ == "__main__":
    main()
