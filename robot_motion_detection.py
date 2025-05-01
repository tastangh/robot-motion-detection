import cv2
import numpy as np
import os

VIDEO_PATH = "odev2-videolar/tusas-odev2-test1.mp4"
OUTPUT_TXT = "odev2-videolar/tusas-odev2-ogr1.txt"
SECONDS = 60
GRID_ROWS, GRID_COLS = 3, 3

FLOW_STD_BOOST = 1.0
MOTION_RATIO = 0.2

camera_matrix = np.array([[800, 0, 320],
                          [0, 800, 240],
                          [0,   0,   1]], dtype=np.float32)
dist_coeffs = np.array([-0.25, 0.1, 0.0, 0.0, 0.0], dtype=np.float32)

txt_order_mapping = {
    0: 7, 1: 1, 2: 4,
    3: 8, 4: 2, 5: 5,
    6: 9, 7: 3, 8: 6
}

def undistort_frame(frame, K, dist):
    h, w = frame.shape[:2]
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(frame, K, dist, None, new_K)
    x, y, w, h = roi
    return dst[y:y+h, x:x+w]

def detect_drop_frame(cap, K, dist):
    history = []
    frame_idx = 0
    stable_threshold = 100000
    diff_peak_threshold = 200000
    frames_to_check_stability = 15

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ret, frame = cap.read()
        if not ret:
            return 0
        frame_undistorted = undistort_frame(frame, K, dist)
        gray = cv2.cvtColor(frame_undistorted, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if history:
            diff = cv2.absdiff(history[-1], gray)
            score = np.sum(diff)
            if score > diff_peak_threshold:
                stable_counter = 0
                current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
                for _ in range(frames_to_check_stability):
                    ret2, f2 = cap.read()
                    if not ret2:
                        break
                    g2 = cv2.cvtColor(undistort_frame(f2, K, dist), cv2.COLOR_BGR2GRAY)
                    g2 = cv2.GaussianBlur(g2, (5, 5), 0)
                    d2 = cv2.absdiff(gray, g2)
                    s2 = np.sum(d2)
                    if s2 < stable_threshold:
                        stable_counter += 1
                    else:
                        break
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
                if stable_counter >= frames_to_check_stability - 2:
                    return frame_idx
        history.append(gray)
        if len(history) > 20:
            history.pop(0)
        frame_idx += 1
        if frame_idx > cap.get(cv2.CAP_PROP_FRAME_COUNT) * 0.25:
            return 0

def split_cells(frame):
    h, w = frame.shape[:2]
    cell_h = h // GRID_ROWS
    cell_w = w // GRID_COLS
    return [(j*cell_w, i*cell_h, (j+1)*cell_w, (i+1)*cell_h)
            for i in range(GRID_ROWS) for j in range(GRID_COLS)]

def get_robot_regions(cells, scale=0.8):
    regions = []
    for (x1, y1, x2, y2) in cells:
        cx, cy = (x1 + x2)//2, (y1 + y2)//2
        w, h = int((x2 - x1) * scale), int((y2 - y1) * scale)
        regions.append((cx - w//2, cy - h//2, cx + w//2, cy + h//2))
    return regions

def main():
    global SECONDS
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("[ERROR] Video açılamadı.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    drop_start = detect_drop_frame(cap, camera_matrix, dist_coeffs)
    if drop_start + int(fps * SECONDS) > total_frames:
        SECONDS = int((total_frames - drop_start) / fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, drop_start)
    ret, first_frame = cap.read()
    if not ret:
        print("[ERROR] İlk frame alınamadı.")
        return

    first_frame = undistort_frame(first_frame, camera_matrix, dist_coeffs)
    cells = split_cells(first_frame)
    robot_regions = get_robot_regions(cells)
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    buffer = [[] for _ in range(9)]
    motion_matrix = np.zeros((SECONDS, 9), dtype=int)

    for frame_idx in range(int(fps * SECONDS)):
        ret, frame = cap.read()
        if not ret:
            break
        frame = undistort_frame(frame, camera_matrix, dist_coeffs)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        second = int(frame_idx / fps)

        for i, (x1, y1, x2, y2) in enumerate(robot_regions):
            roi_prev = prev_gray[y1:y2, x1:x2]
            roi_curr = gray[y1:y2, x1:x2]
            if roi_prev.shape != roi_curr.shape or roi_prev.size == 0:
                continue
            flow = cv2.calcOpticalFlowFarneback(roi_prev, roi_curr, None,
                                                0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            score = np.mean(mag) + FLOW_STD_BOOST * np.std(mag)
            buffer[i].append(score)

            color = (0, 255, 0) if score > 0.15 else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        prev_gray = gray.copy()

        next_sec = int((frame_idx + 1) / fps)
        if next_sec > second or frame_idx == int(fps * SECONDS) - 1:
            for i in range(9):
                scores = buffer[i]
                if scores:
                    mean_score = np.mean(scores)
                    if mean_score > 0.15:
                        motion_matrix[second][i] = 1
            buffer = [[] for _ in range(9)]

        cv2.imshow("Robot Hareket Algılama", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    output_dir = os.path.dirname(OUTPUT_TXT)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

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
