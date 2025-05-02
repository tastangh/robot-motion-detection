import cv2
import numpy as np
import os
import time

VIDEO_PATH = os.getenv("VIDEO_PATH", "odev2-videolar/tusas-odev2-test3.mp4")
OUTPUT_TXT = os.getenv("OUTPUT_TXT", "odev2-videolar/tusas-odev2-ogr3.txt")
SECONDS = 60
GRID_ROWS, GRID_COLS = 3, 3
TARGET_SIZE = (360, 360)

HARRIS_DIFF_THRESHOLD = 9
MOTION_RATIO = 0.2

last_successful_homography = None
txt_order_mapping = {
    0: 7, 1: 1, 2: 4,
    3: 8, 4: 2, 5: 5,
    6: 9, 7: 3, 8: 6
}

display_index_map = {i: txt_order_mapping[i] for i in range(9)}

def get_adaptive_perspective_transform(frame, target_w=TARGET_SIZE[0], target_h=TARGET_SIZE[1]):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    kernel = np.ones((3,3), np.uint8)
    edged = cv2.dilate(edged, kernel, iterations=1)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise RuntimeError("No contours found for rectification")

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    grid_contour = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            if cv2.contourArea(approx) > (frame.shape[0] * frame.shape[1] * 0.1):
                 grid_contour = approx
                 break

    if grid_contour is None:
        raise RuntimeError("Could not find a suitable quadrilateral contour for the grid")

    pts = grid_contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    src_pts = rect
    dst_pts = np.float32([[0, 0], [target_w, 0], [target_w, target_h], [0, target_h]])

    H = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return H

def rectify_frame(frame, target_w=TARGET_SIZE[0], target_h=TARGET_SIZE[1]):
    global last_successful_homography
    try:
        H = get_adaptive_perspective_transform(frame, target_w, target_h)
        if H is None or not np.any(H):
             raise ValueError("Calculated Homography matrix is invalid")
        last_successful_homography = H
    except Exception as e:
        if last_successful_homography is None:
            return None, None
        H = last_successful_homography

    rectified = cv2.warpPerspective(frame, H, (target_w, target_h))
    return rectified, H

def split_cells(frame):
    h, w = frame.shape[:2]
    cell_h = h // GRID_ROWS
    cell_w = w // GRID_COLS
    cells = []
    for i in range(GRID_ROWS):
        for j in range(GRID_COLS):
            y1 = i * cell_h
            x1 = j * cell_w
            y2 = y1 + cell_h
            x2 = x1 + cell_w
            cells.append((y1, y2, x1, x2))
    return cells

def detect_drop_start(video_path, threshold=500000, check_frames=150):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video file: {video_path}")
        return 0
    prev_gray = None
    frame_index = 0
    max_check = check_frames

    while frame_index < max_check:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            motion_score = np.sum(diff)
            if motion_score > threshold:
                print(f"[INFO] Significant motion detected around frame {frame_index}. Assuming drop ended.")
                cap.release()
                # Use cap.get(cv2.CAP_PROP_FRAME_COUNT) within an isOpened check or after ensuring it's valid
                frame_count = 0
                temp_cap = cv2.VideoCapture(video_path)
                if temp_cap.isOpened():
                    frame_count = int(temp_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    temp_cap.release()
                
                if frame_count > 0:
                   return min(frame_index + 5, frame_count - 1)
                else:
                   return frame_index + 5 # Fallback if frame count not available

        prev_gray = gray
        frame_index += 1

    print("[INFO] No significant initial motion detected, starting from beginning.")
    cap.release()
    return 0


def harris_score(gray_roi):
    if gray_roi is None or gray_roi.size == 0: return 0
    dst = cv2.cornerHarris(np.float32(gray_roi), 2, 3, 0.04)
    score = np.sum(dst > 0.01 * dst.max())
    return score

def main():
    global last_successful_homography

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video: {VIDEO_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("[WARN] Could not get FPS, assuming 30.")
        fps = 30

    total_frames_to_process = int(fps * SECONDS)

    drop_end_frame = detect_drop_start(VIDEO_PATH, threshold=300000, check_frames=int(fps*5))
    print(f"[INFO] Starting processing from frame index: {drop_end_frame}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, drop_end_frame)

    motion_matrix = np.zeros((SECONDS, 9), dtype=int)
    frame_motion_buffer = [[] for _ in range(9)]
    prev_harris_scores = [None] * 9

    processed_frame_count = 0
    current_second = 0

    cv2.namedWindow("Robot Motion Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Robot Motion Detection", 720, 720)

    ret, initial_frame = cap.read()
    if ret:
        _, initial_H = rectify_frame(initial_frame)
        if initial_H is not None:
             last_successful_homography = initial_H
             print("[INFO] Initial homography calculated successfully.")
        else:
             print("[WARN] Could not calculate initial homography. Will retry on next frames.")
        cap.set(cv2.CAP_PROP_POS_FRAMES, drop_end_frame)
    else:
        print("[ERROR] Could not read the first frame for initial homography.")
        cap.release()
        return


    while processed_frame_count < total_frames_to_process:
        ret, frame = cap.read()
        if not ret:
            print(f"[WARN] Could not read frame at index {drop_end_frame + processed_frame_count}. End of video?")
            break

        rectified_frame, _ = rectify_frame(frame)

        if rectified_frame is None:
            print(f"[WARN] Skipping frame {drop_end_frame + processed_frame_count} due to rectification failure.")
            processed_frame_count += 1
            continue

        gray_rectified = cv2.cvtColor(rectified_frame, cv2.COLOR_BGR2GRAY)
        cells = split_cells(rectified_frame)

        display_frame = rectified_frame.copy()

        current_frame_has_motion = [0] * 9
        for i, (y1, y2, x1, x2) in enumerate(cells):
            roi_gray = gray_rectified[y1:y2, x1:x2]
            current_score = harris_score(roi_gray)

            motion_detected_in_cell = 0
            if prev_harris_scores[i] is not None:
                diff = abs(current_score - prev_harris_scores[i])
                if diff > HARRIS_DIFF_THRESHOLD:
                    motion_detected_in_cell = 1
                    current_frame_has_motion[i] = 1

            frame_motion_buffer[i].append(motion_detected_in_cell)
            prev_harris_scores[i] = current_score

            color = (0, 255, 0) if motion_detected_in_cell == 1 else (0, 0, 255)
            thickness = 2 if motion_detected_in_cell == 1 else 1
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(display_frame, str(display_index_map[i]), (x1 + 5, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


        processed_frame_count += 1
        if processed_frame_count > 0 and processed_frame_count % int(fps) == 0:
            sec_index = current_second
            if sec_index < SECONDS:
                for i in range(9):
                    if len(frame_motion_buffer[i]) > 0:
                        ratio = sum(frame_motion_buffer[i]) / len(frame_motion_buffer[i])
                        if ratio >= MOTION_RATIO:
                            motion_matrix[sec_index][i] = 1
                        else:
                            motion_matrix[sec_index][i] = 0
                    else:
                         motion_matrix[sec_index][i] = 0

            frame_motion_buffer = [[] for _ in range(9)]
            current_second += 1


        sec_display = int(processed_frame_count / fps) + 1
        cv2.putText(display_frame, f"Sec: {sec_display}", (10, display_frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow("Robot Motion Detection", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Quit key pressed. Exiting.")
            break

    cap.release()
    cv2.destroyAllWindows()

    output_motion_matrix_formatted = np.zeros((SECONDS, 9), dtype=int)
    for internal_idx in range(9):
        txt_col_idx = txt_order_mapping[internal_idx] - 1
        if 0 <= txt_col_idx < 9:
            output_motion_matrix_formatted[:, txt_col_idx] = motion_matrix[:, internal_idx]
        else:
            print(f"[WARN] Invalid mapping for internal index {internal_idx} to Robot # {txt_order_mapping[internal_idx]}")

    print(f"[INFO] Writing motion data to: {OUTPUT_TXT}")
    try:
        with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
            header = "Saniye\t" + "\t".join([f"Robot-{i+1}" for i in range(9)])
            f.write(header + "\n")
            for i in range(SECONDS):
                row_values = "\t".join([f"{output_motion_matrix_formatted[i][j]}" for j in range(9)])
                f.write(f"{i+1:<3}\t{row_values}\n")
        print("[INFO] Output file written successfully.")
    except IOError as e:
        print(f"[ERROR] Could not write output file {OUTPUT_TXT}: {e}")

if __name__ == "__main__":
    main()