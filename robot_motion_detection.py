import cv2
import numpy as np
import os

# === Ayarlar ===
VIDEO_PATH = "odev2-videolar/tusas-odev2-test1.mp4"
OUTPUT_TXT = "odev2-videolar/tusas-odev2-ogr1.txt"
SECONDS = 60
GRID_ROWS, GRID_COLS = 3, 3

HARRIS_DIFF_THRESHOLD = 7
MOTION_RATIO = 0.2  # %20

txt_order_mapping = {
    0: 7, 1: 1, 2: 4,
    3: 8, 4: 2, 5: 5,
    6: 9, 7: 3, 8: 6
}
display_index_map = txt_order_mapping

def fixed_perspective_rectify(frame):
    h, w = frame.shape[:2]
    src_pts = np.float32([
        [70, 50],
        [w - 70, 50],
        [w - 70, h - 60],
        [70, h - 60]
    ])
    dst_pts = np.float32([
        [0, 0],
        [360, 0],
        [360, 360],
        [0, 360]
    ])
    H = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(frame, H, (360, 360))

def split_cells(frame):
    h, w = frame.shape[:2]
    cell_h = h // GRID_ROWS
    cell_w = w // GRID_COLS
    cells = []
    for i in range(GRID_ROWS):
        for j in range(GRID_COLS):
            x1 = j * cell_w
            y1 = i * cell_h
            x2 = x1 + cell_w
            y2 = y1 + cell_h
            cells.append((x1, y1, x2, y2))
    return cells

def detect_drop_start(video_path, threshold=500000):
    cap = cv2.VideoCapture(video_path)
    prev_gray = None
    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            motion_score = np.sum(diff)
            if motion_score > threshold:
                cap.release()
                return frame_index
        prev_gray = gray
        frame_index += 1
    cap.release()
    return 0

def harris_score(gray):
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)
    dst = cv2.cornerHarris(np.float32(blurred), 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    corners = dst > 0.01 * dst.max()
    return np.sum(corners)

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    drop_start = detect_drop_start(VIDEO_PATH)
    print(f"[INFO] Robot düşme sonrası başlangıç frame: {drop_start}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, drop_start)

    ret, first_frame = cap.read()
    if not ret:
        print("[ERROR] İlk frame alınamadı.")
        return

    first_frame = fixed_perspective_rectify(first_frame)
    cells = split_cells(first_frame)

    motion_matrix = np.zeros((SECONDS, 9), dtype=int)
    frame_buffer = [[] for _ in range(9)]
    prev_scores = [None for _ in range(9)]
    frame_idx = 0

    cv2.namedWindow("Robot Harris Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Robot Harris Detection", 720, 720)

    while frame_idx < int(fps * SECONDS):
        ret, frame = cap.read()
        if not ret:
            break
        frame = fixed_perspective_rectify(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        second = int(frame_idx / fps)

        for i, (x1, y1, x2, y2) in enumerate(cells):
            roi = gray[y1:y2, x1:x2]
            score = harris_score(roi)
            if prev_scores[i] is not None:
                diff = abs(score - prev_scores[i])
                frame_buffer[i].append(1 if diff > HARRIS_DIFF_THRESHOLD else 0)
            prev_scores[i] = score

        if frame_idx % int(fps) == 0 and frame_idx > 0:
            sec = int(frame_idx / fps) - 1
            for i in range(9):
                ratio = sum(frame_buffer[i]) / len(frame_buffer[i])
                if ratio >= MOTION_RATIO:
                    motion_matrix[sec][i] = 1
            frame_buffer = [[] for _ in range(9)]

        for i, (x1, y1, x2, y2) in enumerate(cells):
            if len(frame_buffer[i]) > 0 and frame_buffer[i][-1] == 1:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.putText(frame, str(display_index_map[i]), (x1 + 5, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Robot Harris Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    # TXT çıktısı (istenen formatta)
    output_motion_matrix = np.zeros((SECONDS, 9), dtype=int)
    for internal_idx in range(9):
        txt_col_idx = txt_order_mapping[internal_idx] - 1
        output_motion_matrix[:, txt_col_idx] = motion_matrix[:, internal_idx]

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("Saniye\tRobot-1 Robot-2 Robot-3 Robot-4 Robot-5 Robot-6 Robot-7 Robot-8 Robot-9\n")
        for i in range(SECONDS):
            row = "\t".join([f"{output_motion_matrix[i][j]:>4}" for j in range(9)])
            f.write(f"{i+1:3})\t{row}\n")

    print(f"[INFO] Hareket verisi yazıldı: {OUTPUT_TXT}")

if __name__ == "__main__":
    main()
