import cv2
import numpy as np

VIDEO_PATH = 'tusas-odev2-test1.mp4'
OUTPUT_TXT = 'tusas-odev2-ogr.txt'
NUM_SECONDS = 60
NUM_ROBOTS = 9
MOTION_THRESHOLD_RATIO = 0.2

ROBOT_THRESHOLDS = [
    0.0010, 0.0012, 0.0012,
    0.0012, 0.0012, 0.0010,
    0.0008, 0.0011, 0.0009
]

OUTPUT_ORDER = [1, 4, 7, 2, 5, 8, 0, 3, 6]


def rectify_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    corners = cv2.goodFeaturesToTrack(gray_blur, maxCorners=100, qualityLevel=0.01, minDistance=30)
    if corners is None or len(corners) < 4:
        print("[WARN] Yeterli köşe bulunamadı!")
        return frame

    corners = corners.astype(np.intp).reshape(-1, 2)

    # Sadece dış dört köşe ile transform yap
    rect = np.zeros((4, 2), dtype="float32")
    s = corners.sum(axis=1)
    rect[0] = corners[np.argmin(s)]  # top-left
    rect[2] = corners[np.argmax(s)]  # bottom-right

    diff = np.diff(corners, axis=1)
    rect[1] = corners[np.argmin(diff)]  # top-right
    rect[3] = corners[np.argmax(diff)]  # bottom-left

    dst = np.array([
        [0, 0],
        [900, 0],
        [900, 900],
        [0, 900]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(frame, M, (900, 900))
    return warped


def get_grid_cells(frame):
    h, w = frame.shape[:2]
    ch, cw = h // 3, w // 3
    return [(j * cw, i * ch, (j + 1) * cw, (i + 1) * ch) for i in range(3) for j in range(3)]


def detect_drop_start(video_path, threshold=500000, check_frames=150):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video file: {video_path}")
        return 0
    prev_gray = None
    frame_index = 0

    while frame_index < check_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = rectify_frame(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            motion_score = np.sum(diff)
            if motion_score > threshold:
                print(f"[INFO] Düşüş sonrası büyük hareket {frame_index}. karede tespit edildi.")
                cap.release()
                frame_count = 0
                temp_cap = cv2.VideoCapture(video_path)
                if temp_cap.isOpened():
                    frame_count = int(temp_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    temp_cap.release()
                return min(frame_index + 5, frame_count - 1) if frame_count else frame_index + 5

        prev_gray = gray
        frame_index += 1

    print("[INFO] Belirgin bir düşme hareketi tespit edilemedi, baştan başlanacak.")
    cap.release()
    return 0


def detect_motion(prev_gray, curr_gray, epsilon):
    diff = cv2.absdiff(prev_gray, curr_gray)
    _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
    motion_pixels = np.sum(thresh) / 255
    area = prev_gray.shape[0] * prev_gray.shape[1]
    motion_ratio = motion_pixels / area
    return motion_ratio > epsilon


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("❌ Video açılamadı.")
        return

    FPS = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = NUM_SECONDS * FPS
    print(f"[INFO] FPS: {FPS}")

    fall_frame = detect_drop_start(VIDEO_PATH)
    start_frame = max(0, fall_frame - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    ret, prev_frame = cap.read()
    if not ret:
        print("❌ Başlangıç karesi alınamadı.")
        return

    prev_frame = rectify_frame(prev_frame)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    cells = get_grid_cells(prev_gray)
    prev_rois = [prev_gray[y1:y2, x1:x2] for (x1, y1, x2, y2) in cells]
    motion_log = [[] for _ in range(NUM_ROBOTS)]

    frame_idx = 1
    while frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = rectify_frame(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for idx, (x1, y1, x2, y2) in enumerate(cells):
            roi = gray[y1:y2, x1:x2]
            moved = detect_motion(prev_rois[idx], roi, epsilon=ROBOT_THRESHOLDS[idx])
            motion_log[idx].append(1 if moved else 0)
            if moved:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            prev_rois[idx] = roi

        # cv2.imshow("Motion Detection", frame)
        # if cv2.waitKey(1) & 0xFF == 27:
        #     break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

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
