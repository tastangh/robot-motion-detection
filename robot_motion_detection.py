import cv2
import numpy as np
import os

VIDEO_PATH = os.getenv("VIDEO_PATH", "odev2-videolar/tusas-odev2-test1.mp4")
OUTPUT_TXT = os.getenv("OUTPUT_TXT", "odev2-videolar/tusas-odev2-ogr1.txt")
SECONDS = 60
GRID_ROWS, GRID_COLS = 3, 3
TARGET_SIZE = (360, 360)
HARRIS_DIFF_THRESHOLD = 6
MOTION_RATIO = 0.2

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
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return last_successful_homography

    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            rect = approx.reshape(4, 2)
            s = rect.sum(axis=1)
            diff = np.diff(rect, axis=1)
            ordered = np.array([
                rect[np.argmin(s)],
                rect[np.argmin(diff)],
                rect[np.argmax(s)],
                rect[np.argmax(diff)],
            ], dtype="float32")
            dst = np.float32([[0,0], [TARGET_SIZE[0],0], [TARGET_SIZE[0],TARGET_SIZE[1]], [0,TARGET_SIZE[1]]])
            H = cv2.getPerspectiveTransform(ordered, dst)
            last_successful_homography = H
            return H
    return last_successful_homography

def rectify(frame):
    H = get_homography(frame)
    if H is None:
        return None
    return cv2.warpPerspective(frame, H, TARGET_SIZE)

def split_grid(frame):
    h, w = frame.shape[:2]
    ch, cw = h // GRID_ROWS, w // GRID_COLS
    return [(r*ch, (r+1)*ch, c*cw, (c+1)*cw) for r in range(GRID_ROWS) for c in range(GRID_COLS)]

def harris_score(gray):
    if gray is None or gray.size == 0: return 0
    dst = cv2.cornerHarris(np.float32(gray), 2, 3, 0.04)
    if dst.max() < 1e-5:
        return 0
    return np.sum(dst > 0.01 * dst.max())

def main():
    global last_successful_homography
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[ERROR] {VIDEO_PATH} açılamadı.")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(fps * SECONDS)
    drop_frame = detect_drop_start(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_POS_FRAMES, drop_frame)

    motion_matrix = np.zeros((SECONDS, 9), dtype=int)
    buffers = [[] for _ in range(9)]
    prev_scores = [None]*9
    second = 0
    count = 0

    ret, f0 = cap.read()
    if ret:
        rect = rectify(f0)
        if rect is not None:
            print("[INFO] İlk homografi başarıyla hesaplandı.")
        cap.set(cv2.CAP_PROP_POS_FRAMES, drop_frame)

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
        display = rect.copy()
        motions = [0]*9
        for i, (y1,y2,x1,x2) in enumerate(cells):
            roi = gray[y1:y2, x1:x2]
            score = harris_score(roi)
            diff = abs(score - prev_scores[i]) if prev_scores[i] else 0
            if diff > HARRIS_DIFF_THRESHOLD:
                motions[i] = 1
            prev_scores[i] = score
            buffers[i].append(motions[i])
            color = (0,255,0) if motions[i] else (0,0,255)
            cv2.rectangle(display, (x1,y1), (x2,y2), color, 2 if motions[i] else 1)
            cv2.putText(display, str(display_index_map[i]), (x1+5, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        if count % int(fps) == int(fps)-1:
            for i in range(9):
                ratio = sum(buffers[i]) / len(buffers[i]) if buffers[i] else 0
                motion_matrix[second][i] = 1 if ratio >= MOTION_RATIO else 0
            buffers = [[] for _ in range(9)]
            second += 1

        cv2.putText(display, f"Sec: {second+1}", (10, display.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1)
        cv2.imshow("Robot Motion Detection", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        count += 1

    cap.release()
    cv2.destroyAllWindows()

    print(f"[INFO] Yazılıyor: {OUTPUT_TXT}")
    final_matrix = np.zeros((SECONDS, 9), dtype=int)
    for i in range(9):
        col = txt_order_mapping[i] - 1
        final_matrix[:, col] = motion_matrix[:, i]
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("Saniye\t" + "\t".join([f"Robot-{i+1}" for i in range(9)]) + "\n")
        for i in range(SECONDS):
            f.write(f"{i+1:<3}\t" + "\t".join(map(str, final_matrix[i])) + "\n")
    print("[✅] Tamamlandı.")

    # === Otomatik Karşılaştırma ===
    ref_path = OUTPUT_TXT.replace("ogr", "referans")
    if os.path.exists(ref_path):
        try:
            from compare_outputs import compare_outputs
            print(f"\n[INFO] Karşılaştırma başlatılıyor: {ref_path} vs {OUTPUT_TXT}")
            compare_outputs(ref_path, OUTPUT_TXT)
        except Exception as e:
            print(f"[HATA] compare_outputs import edilemedi: {e}")
    else:
        print(f"[UYARI] Referans dosyası bulunamadı: {ref_path}")

if __name__ == "__main__":
    main()
