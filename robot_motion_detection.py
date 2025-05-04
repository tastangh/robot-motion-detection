import cv2
import numpy as np
import os

VIDEO_PATH = os.getenv("VIDEO_PATH", "odev2-videolar/tusas-odev2-test1.mp4")
OUTPUT_TXT = os.getenv("OUTPUT_TXT", "odev2-videolar/tusas-odev2-ogr1.txt")
SECONDS = 60
EVAL_INTERVAL = 2  # saniyelik karar penceresi
FPS_FALLBACK = 30
GRID_ROWS, GRID_COLS = 3, 3
TARGET_SIZE = (360, 360)

HARRIS_BASE_THRESHOLD = 0.12
VOTE_WINDOW = 2
MOTION_RATIO = 0.10


txt_order_mapping = {0: 7, 1: 1, 2: 4, 3: 8, 4: 2, 5: 5, 6: 9, 7: 3, 8: 6}
display_index_map = {i: txt_order_mapping[i] for i in range(9)}

last_successful_homography = None

def detect_drop_start(video_path, threshold=300000):
    cap = cv2.VideoCapture(video_path)
    prev = None
    for i in range(150):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        if prev is not None:
            diff = cv2.absdiff(prev, gray)
            score = np.sum(diff)
            if score > threshold:
                cap.release()
                return i + 5
        prev = gray
    cap.release()
    return 0

def get_homography(frame):
    global last_successful_homography
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            s = pts.sum(axis=1)
            diff = np.diff(pts, axis=1)
            ordered = np.array([
                pts[np.argmin(s)],
                pts[np.argmin(diff)],
                pts[np.argmax(s)],
                pts[np.argmax(diff)],
            ], dtype="float32")
            dst = np.float32([[0,0], [TARGET_SIZE[0],0], [TARGET_SIZE[0],TARGET_SIZE[1]], [0,TARGET_SIZE[1]]])
            H = cv2.getPerspectiveTransform(ordered, dst)
            last_successful_homography = H
            return H
    return last_successful_homography

def rectify(frame):
    H = get_homography(frame)
    if H is None: return None
    return cv2.warpPerspective(frame, H, TARGET_SIZE)

def split_grid(frame):
    h, w = frame.shape[:2]
    ch, cw = h // GRID_ROWS, w // GRID_COLS
    return [(r*ch, (r+1)*ch, c*cw, (c+1)*cw) for r in range(GRID_ROWS) for c in range(GRID_COLS)]

def harris_score(gray):
    dst = cv2.cornerHarris(np.float32(gray), 2, 3, 0.04)
    return np.sum(dst > 0.01 * dst.max()) if dst.max() > 1e-5 else 0

def main():
    global last_successful_homography
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("[ERR] Video açılamadı.")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS_FALLBACK
    total_frames = int(fps * SECONDS)
    interval_frames = int(fps * EVAL_INTERVAL)

    drop_frame = detect_drop_start(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_POS_FRAMES, drop_frame)

    motion_matrix = np.zeros((SECONDS, 9), dtype=int)
    vote_history = [[0]*VOTE_WINDOW for _ in range(9)]
    prev_scores = [None]*9
    count = 0

    _, f0 = cap.read()
    if f0 is not None:
        rectify(f0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, drop_frame)

    frame_buffer = [[] for _ in range(9)]

    while count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        rect = rectify(frame)
        if rect is None:
            count += 1
            continue

        gray = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY)
        cells = split_grid(rect)
        current_motion = [0]*9

        for i, (y1,y2,x1,x2) in enumerate(cells):
            roi = gray[y1:y2, x1:x2]
            score = harris_score(roi)
            prev = prev_scores[i] if prev_scores[i] else 1e-5
            diff_ratio = abs(score - prev) / (prev + 1e-5)
            prev_scores[i] = score
            current_motion[i] = 1 if diff_ratio > HARRIS_BASE_THRESHOLD else 0
            frame_buffer[i].append(current_motion[i])

        if count % interval_frames == interval_frames - 1:
            for i in range(9):
                ratio = sum(frame_buffer[i]) / len(frame_buffer[i])
                motion_vote = 1 if ratio >= MOTION_RATIO else 0
                vote_history[i].append(motion_vote)
                vote_history[i] = vote_history[i][-VOTE_WINDOW:]
                final_vote = 1 if sum(vote_history[i]) >= 1 else 0
                second_idx = int(count / fps)
                if second_idx < SECONDS:
                    motion_matrix[second_idx][i] = final_vote
            frame_buffer = [[] for _ in range(9)]
        count += 1

    final_matrix = np.zeros((SECONDS, 9), dtype=int)
    for i in range(9):
        col = txt_order_mapping[i] - 1
        final_matrix[:, col] = motion_matrix[:, i]

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("Saniye\t" + "\t".join([f"Robot-{i+1}" for i in range(9)]) + "\n")
        for i in range(SECONDS):
            f.write(f"{i+1:<3}\t" + "\t".join(map(str, final_matrix[i])) + "\n")
    print(f"[✅] Yazıldı: {OUTPUT_TXT}")

    ref_path = OUTPUT_TXT.replace("ogr", "referans")
    if os.path.exists(ref_path):
        try:
            from compare_outputs import compare_outputs
            print(f"\n[INFO] Karşılaştırma başlatılıyor: {ref_path} vs {OUTPUT_TXT}")
            compare_outputs(ref_path, OUTPUT_TXT)
        except Exception as e:
            print(f"[HATA] compare_outputs içe aktarılamadı: {e}")
    else:
        print(f"[UYARI] Referans dosyası yok: {ref_path}")

if __name__ == "__main__":
    main()