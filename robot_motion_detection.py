import cv2
import numpy as np
import os

# === Ayarlar ===
VIDEO_PATH = "odev2-videolar/tusas-odev2-test1.mp4"
OUTPUT_TXT = "odev2-videolar/tusas-odev2-ogr1.txt"
SECONDS = 60
GRID_ROWS, GRID_COLS = 3, 3

FLOW_STD_BOOST = 1.0
MOTION_RATIO = 0.2
PADDING_RATIO = 0.1

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
    cell_h = h / GRID_ROWS
    cell_w = w / GRID_COLS
    cells = []
    for i in range(GRID_ROWS):
        for j in range(GRID_COLS):
            x1 = int(round(j * cell_w))
            y1 = int(round(i * cell_h))
            x2 = int(round((j + 1) * cell_w))
            y2 = int(round((i + 1) * cell_h))
            cells.append((x1, y1, x2, y2))
    return cells

def get_robot_regions(cells, padding_ratio=0.1):
    regions = []
    for (x1, y1, x2, y2) in cells:
        w = x2 - x1
        h = y2 - y1
        pad_w = int(w * padding_ratio)
        pad_h = int(h * padding_ratio)
        regions.append((x1 + pad_w, y1 + pad_h, x2 - pad_w, y2 - pad_h))
    return regions

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("[ERROR] Video açılamadı.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    ret, first_frame = cap.read()
    if not ret:
        print("[ERROR] İlk frame alınamadı.")
        return

    first_frame = fixed_perspective_rectify(first_frame)
    cells = split_cells(first_frame)
    robot_regions = get_robot_regions(cells, padding_ratio=PADDING_RATIO)
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    buffer = [[] for _ in range(9)]
    motion_matrix = np.zeros((SECONDS, 9), dtype=int)

    for frame_idx in range(int(fps * SECONDS)):
        ret, frame = cap.read()
        if not ret:
            break
        frame = fixed_perspective_rectify(frame)
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
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            cv2.putText(frame, str(display_index_map[i]), (x1 + 5, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

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