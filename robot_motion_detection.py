import cv2
import numpy as np
import os
import time

VIDEO_PATH = os.getenv("VIDEO_PATH", "odev2-videolar/tusas-odev2-test1.mp4")
OUTPUT_TXT = os.getenv("OUTPUT_TXT", "odev2-videolar/tusas-odev2-ogr1.txt")
SECONDS = 60
GRID_ROWS, GRID_COLS = 3, 3
TARGET_SIZE = (360, 360)

HARRIS_BLOCK_SIZE = 2
HARRIS_KSIZE = 3
HARRIS_K = 0.06
HARRIS_THRESHOLD_RATIO = 0.02
HARRIS_DIFF_THRESHOLD = 3
APPLY_BLUR_BEFORE_HARRIS = False
HARRIS_BLUR_KERNEL = (3, 3)

MOTION_SECOND_RATIO = 0.20

txt_order_mapping = {0: 7, 1: 1, 2: 4, 3: 8, 4: 2, 5: 5, 6: 9, 7: 3, 8: 6}
display_index_map = {i: txt_order_mapping[i] for i in range(9)}

last_successful_homography = None


def detect_drop_start(video_path, threshold=300000):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Video {video_path} açılamadı.")
        return 0
    prev_gray = None
    frame_count = 0
    max_frames_to_check = 200

    print("[INFO] Başlangıç karesi tespiti...")
    while frame_count < max_frames_to_check:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            score = np.sum(diff)
            if score > threshold:
                cap.release()
                start_frame = frame_count + 5
                print(f"[INFO] Belirgin hareket {frame_count}. kare civarında. Referans noktası: {start_frame}.")
                return start_frame
        prev_gray = gray

    cap.release()
    print("[INFO] Başlangıçta belirgin hareket yok, 0. kare referans noktası.")
    return 0

def get_homography(frame):
    global last_successful_homography
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return last_successful_homography

    largest_area = 0
    best_approx = None

    for c in contours:
        area = cv2.contourArea(c)
        min_area_threshold = TARGET_SIZE[0] * TARGET_SIZE[1] * 0.05
        if area < min_area_threshold:
             continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            (x, y, w, h) = cv2.boundingRect(approx)
            if h == 0: continue
            aspect_ratio = w / float(h)
            if 0.7 < aspect_ratio < 1.3:
                if area > largest_area:
                    largest_area = area
                    best_approx = approx

    if best_approx is not None:
            rect = best_approx.reshape(4, 2)
            s = rect.sum(axis=1)
            diff = np.diff(rect, axis=1)
            ordered_pts = np.zeros((4, 2), dtype="float32")

            ordered_pts[0] = rect[np.argmin(s)]
            ordered_pts[2] = rect[np.argmax(s)]
            ordered_pts[1] = rect[np.argmin(diff)]
            ordered_pts[3] = rect[np.argmax(diff)]

            dst_pts = np.array([
                [0, 0],
                [TARGET_SIZE[0] - 1, 0],
                [TARGET_SIZE[0] - 1, TARGET_SIZE[1] - 1],
                [0, TARGET_SIZE[1] - 1]], dtype="float32")

            H = cv2.getPerspectiveTransform(ordered_pts, dst_pts)
            last_successful_homography = H
            return H
    else:
        return last_successful_homography

def rectify(frame):
    H = get_homography(frame)
    if H is None:
        return None
    return cv2.warpPerspective(frame, H, TARGET_SIZE)

def split_grid(frame_shape):
    h, w = frame_shape[:2]
    cell_h, cell_w = h // GRID_ROWS, w // GRID_COLS
    grid_coords = []
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            y1 = r * cell_h
            y2 = (r + 1) * cell_h
            x1 = c * cell_w
            x2 = (c + 1) * cell_w
            grid_coords.append((y1, y2, x1, x2))
    return grid_coords

def calculate_harris_score(gray_roi):
    if gray_roi is None or gray_roi.size == 0:
        return 0

    processed_roi = gray_roi
    if APPLY_BLUR_BEFORE_HARRIS and HARRIS_BLUR_KERNEL is not None:
        processed_roi = cv2.GaussianBlur(gray_roi, HARRIS_BLUR_KERNEL, 0)

    gray_roi_float = np.float32(processed_roi)
    dst = cv2.cornerHarris(gray_roi_float, HARRIS_BLOCK_SIZE, HARRIS_KSIZE, HARRIS_K)

    max_val = dst.max()
    if max_val <= 1e-6:
        return 0

    corner_count = np.sum(dst > HARRIS_THRESHOLD_RATIO * max_val)
    return corner_count


def main():
    global last_successful_homography
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[ERROR] Video {VIDEO_PATH} açılamadı.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or fps > 120:
        print(f"[WARN] FPS okunamadı ({fps}). 30 FPS varsayılıyor.")
        fps = 30.0
    else:
        print(f"[INFO] Video FPS: {fps}")
    frame_interval_ms = int(1000 / fps)
    one_second_frames = int(round(fps))

    total_frames_to_process = int(fps * SECONDS)
    drop_ref_frame_index = detect_drop_start(VIDEO_PATH)

    # Calculate the effective start frame: drop reference + 1 second
    effective_start_frame = drop_ref_frame_index + one_second_frames
    print(f"[INFO] 1 saniye ({one_second_frames} kare) ekleniyor.")


    if effective_start_frame > 0:
        set_success = cap.set(cv2.CAP_PROP_POS_FRAMES, effective_start_frame)
        current_frame_check = cap.get(cv2.CAP_PROP_POS_FRAMES)
        # Check if seek was successful, allow slight inaccuracy
        if not set_success or abs(current_frame_check - effective_start_frame) > 1:
             print(f"[WARN] Kareye gitme başarısız ({effective_start_frame} istendi, {current_frame_check} alındı).")
        # Update effective_start_frame to where we actually landed
        effective_start_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        print(f"[INFO] Analiz başlangıç karesi ayarlandı: {effective_start_frame}")
    else:
         # This case likely won't happen if drop_ref_frame_index >= 0 and fps > 0
         print(f"[INFO] Analiz 0. kareden başlıyor (Hesaplanan başlangıç: {effective_start_frame}).")
         effective_start_frame = 0
         cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Ensure position is 0


    motion_matrix = np.zeros((SECONDS, 9), dtype=int)
    frame_motion_buffers = [[] for _ in range(9)]
    prev_harris_scores = [None] * 9
    grid_cell_coords = split_grid(TARGET_SIZE)

    second_counter = 0
    frame_counter_in_second = 0
    processed_frame_count = 0
    rectification_failures = 0

    print("[INFO] İlk homografi deneniyor...")
    initial_homography_found = False
    temp_cap = cv2.VideoCapture(VIDEO_PATH)
    if temp_cap.isOpened():
        temp_cap.set(cv2.CAP_PROP_POS_FRAMES, effective_start_frame)
        for _ in range(int(fps) * 2):
            ret_h, frame_h = temp_cap.read()
            if not ret_h: break
            if rectify(frame_h) is not None:
                 print("[INFO] İlk homografi bulundu.")
                 initial_homography_found = True
                 break
        temp_cap.release()
    else:
        print("[WARN] İlk homografi için geçici video okuyucu açılamadı.")

    if not initial_homography_found:
         print("[WARN] İlk homografi bulunamadı.")

    print(f"[INFO] İlk referans kare ({effective_start_frame}) işleniyor...")
    ret, first_frame = cap.read()
    initial_frame_processed = False
    if ret:
        rectified_first = rectify(first_frame)
        if rectified_first is not None:
            print("[INFO] İlk kare başarıyla düzeltildi.")
            gray_first = cv2.cvtColor(rectified_first, cv2.COLOR_BGR2GRAY)
            temp_scores = [0] * 9
            for i, (y1, y2, x1, x2) in enumerate(grid_cell_coords):
                gray_roi = gray_first[y1:y2, x1:x2]
                score = calculate_harris_score(gray_roi)
                temp_scores[i] = score
            prev_harris_scores = temp_scores
            initial_frame_processed = True
            print(f"[INFO] İlk kare Harris skorları hesaplandı: {prev_harris_scores}")
        else:
            print("[WARN] İlk referans kare düzeltilemedi. İlk karşılaştırma yapılamayacak.")
    else:
        print("[ERROR] İlk referans kare okunurken video bitti.")
        cap.release()
        return

    while processed_frame_count < total_frames_to_process:
        start_time = time.time()
        ret, frame = cap.read()
        current_loop_frame_index = cap.get(cv2.CAP_PROP_POS_FRAMES) -1

        if not ret:
            print("[INFO] Video sonuna ulaşıldı.")
            break

        rectified_frame = rectify(frame)

        if rectified_frame is None:
            rectification_failures += 1
            if rectification_failures > 0 and rectification_failures % int(fps) == 0:
                 print(f"[WARN] Düzeltme {rectification_failures} kez başarısız oldu (son 1sn).")
            processed_frame_count += 1
            frame_counter_in_second += 1
        else:
            rectification_failures = 0
            gray_rect = cv2.cvtColor(rectified_frame, cv2.COLOR_BGR2GRAY)
            display_frame = rectified_frame.copy()
            current_frame_motions = [0] * 9
            current_harris_scores = [0] * 9

            for i, (y1, y2, x1, x2) in enumerate(grid_cell_coords):
                gray_roi = gray_rect[y1:y2, x1:x2]
                score = calculate_harris_score(gray_roi)
                current_harris_scores[i] = score
                score_diff = 0

                if prev_harris_scores[i] is not None:
                    score_diff = abs(score - prev_harris_scores[i])
                    if score_diff > HARRIS_DIFF_THRESHOLD:
                        current_frame_motions[i] = 1

                color = (0, 255, 0) if current_frame_motions[i] == 1 else (0, 0, 255)
                thickness = 2 if current_frame_motions[i] == 1 else 1
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(display_frame, str(display_index_map[i]),
                            (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            prev_harris_scores = current_harris_scores

            for i in range(9):
                 frame_motion_buffers[i].append(current_frame_motions[i])

            cv2.putText(display_frame, f"Saniye: {second_counter + 1}/{SECONDS}", (10, display_frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            cv2.putText(display_frame, f"Kare: {int(current_loop_frame_index)} (islenen: {processed_frame_count+1})", (10, display_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            cv2.imshow("Robot Hareket Tespiti", display_frame)

            processed_frame_count += 1
            frame_counter_in_second += 1

        if frame_counter_in_second >= int(round(fps)):
            if second_counter < SECONDS:
                for i in range(9):
                     buffer = frame_motion_buffers[i]
                     if buffer:
                         ratio = sum(buffer) / len(buffer)
                         motion_matrix[second_counter][i] = 1 if ratio >= MOTION_SECOND_RATIO else 0
                     else:
                         motion_matrix[second_counter][i] = 0

            frame_motion_buffers = [[] for _ in range(9)]
            second_counter += 1
            frame_counter_in_second = 0
            if second_counter >= SECONDS:
                print(f"[INFO] Hedef süre ({SECONDS} sn) tamamlandı.")
                break

        elapsed_time_ms = (time.time() - start_time) * 1000
        wait_time = max(1, frame_interval_ms - int(elapsed_time_ms))
        key = cv2.waitKey(wait_time) & 0xFF
        if key == ord('q'):
            print("[INFO] Kullanıcı isteğiyle çıkılıyor ('q').")
            break
        elif key == ord('p'):
             print("[INFO] Duraklatıldı. Devam etmek için bir tuşa basın.")
             cv2.waitKey(0)
             print("[INFO] Devam ediliyor.")

    cap.release()
    cv2.destroyAllWindows()

    print(f"[INFO] Sonuçlar {min(second_counter, SECONDS)} saniye için yazılıyor: {OUTPUT_TXT}")
    final_output_matrix = np.zeros((SECONDS, 9), dtype=int)
    processed_seconds = min(second_counter, SECONDS)

    for internal_idx in range(9):
        target_col_idx = txt_order_mapping[internal_idx] - 1
        if processed_seconds > 0:
            final_output_matrix[:processed_seconds, target_col_idx] = motion_matrix[:processed_seconds, internal_idx]

    try:
        with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
            f.write("Saniye\t" + "\t".join([f"Robot-{i+1}" for i in range(9)]) + "\n")
            for i in range(SECONDS):
                f.write(f"{i+1:<6}\t" + "\t".join(map(str, final_output_matrix[i])) + "\n")
        print("[✅] Yazma tamamlandı.")
    except IOError as e:
        print(f"[ERROR] Çıktı dosyasına yazılamadı {OUTPUT_TXT}: {e}")

    ref_path = OUTPUT_TXT.replace("ogr", "referans")
    if os.path.exists(ref_path):
        print("-" * 30)
        try:
            from compare_outputs import compare_outputs
            print(f"[INFO] Karşılaştırma başlıyor: {ref_path} vs {OUTPUT_TXT}")
            compare_outputs(ref_path, OUTPUT_TXT)
        except ImportError:
            print("[HATA] 'compare_outputs' import edilemedi.")
            print("       Otomatik karşılaştırma atlanıyor.")
        except Exception as e:
            print(f"[HATA] Karşılaştırma sırasında hata: {e}")
        print("-" * 30)
    else:
        print(f"[UYARI] Referans dosyası bulunamadı: {ref_path}. Karşılaştırma atlanıyor.")

if __name__ == "__main__":
    main()