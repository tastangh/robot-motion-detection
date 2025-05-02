import cv2
import numpy as np
import os
import time # For potential debugging/timing

VIDEO_PATH = os.getenv("VIDEO_PATH", "odev2-videolar/tusas-odev2-test3.mp4") # Corrected path based on image text
OUTPUT_TXT = os.getenv("OUTPUT_TXT", "odev2-videolar/tusas-odev2-test3-ogr.txt") # Adjusted output name
SECONDS = 60
GRID_ROWS, GRID_COLS = 3, 3
TARGET_SIZE = (360, 360) # Define target rectified size

HARRIS_DIFF_THRESHOLD = 9 # Threshold for Harris score difference to count as motion
MOTION_RATIO = 0.2  # Minimum ratio of 'motion' frames within a second to mark the second as 'motion' (20%)

# This will store the last successfully calculated homography
last_successful_homography = None
txt_order_mapping = {
    0: 7, 1: 1, 2: 4,
    3: 8, 4: 2, 5: 5,
    6: 9, 7: 3, 8: 6
}
# Mapping for the output TXT file columns (Robot 1-9) based on grid cell index (0-8)
# Grid index (row-major):
# 0 1 2
# 3 4 5
# 6 7 8
# Output Robot number desired for each index:

# Let's create the map used for displaying numbers on screen (can be different if needed)
# Often helpful to display the grid index 0-8 or the target robot number 1-9
display_index_map = {i: txt_order_mapping[i] for i in range(9)} # Display target Robot number

# --- Improved Perspective Transform Function ---
def get_adaptive_perspective_transform(frame, target_w=TARGET_SIZE[0], target_h=TARGET_SIZE[1]):
    """
    Tries to find the main grid contour and calculate the perspective transform.
    Uses contour finding which might be more robust for grid structures.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Adjust Canny thresholds if needed based on video lighting/contrast
    edged = cv2.Canny(blurred, 50, 150)
    # Dilate edges slightly to help connect broken lines of the grid
    kernel = np.ones((3,3), np.uint8)
    edged = cv2.dilate(edged, kernel, iterations=1)


    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise RuntimeError("No contours found for rectification")

    # Sort contours by area, descending, and find the largest quadrilateral
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    grid_contour = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True) # Adjust epsilon (0.02) if needed

        if len(approx) == 4:
            # Check if the contour area is reasonably large (e.g., > 10% of image area)
            # This helps avoid picking small noise contours
            if cv2.contourArea(approx) > (frame.shape[0] * frame.shape[1] * 0.1):
                 grid_contour = approx
                 # Draw the detected contour for debugging
                 # cv2.drawContours(frame, [grid_contour], -1, (0, 255, 0), 2)
                 break

    if grid_contour is None:
        # Optional: Add a fallback to the original goodFeaturesToTrack method here if needed
        raise RuntimeError("Could not find a suitable quadrilateral contour for the grid")

    # Ensure points are in the correct order: TL, TR, BR, BL
    pts = grid_contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # Top-left
    rect[2] = pts[np.argmax(s)] # Bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # Top-right
    rect[3] = pts[np.argmax(diff)] # Bottom-left

    src_pts = rect
    dst_pts = np.float32([[0, 0], [target_w, 0], [target_w, target_h], [0, target_h]])

    H = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return H

# --- Rectify Function (Applies transform) ---
def rectify_frame(frame, target_w=TARGET_SIZE[0], target_h=TARGET_SIZE[1]):
    """
    Calculates and applies the perspective transform for the given frame.
    Returns the rectified frame and the calculated homography matrix.
    Uses the last known good homography as a fallback.
    """
    global last_successful_homography
    try:
        H = get_adaptive_perspective_transform(frame, target_w, target_h)
        # Sanity check the matrix (optional but good practice)
        if H is None or not np.any(H):
             raise ValueError("Calculated Homography matrix is invalid")
        last_successful_homography = H # Store the good matrix
        # print("[DEBUG] Successfully calculated new homography.") # Debug message
    except Exception as e:
        # print(f"[WARN] Rectification failed for this frame: {e}. Using last known transform.")
        if last_successful_homography is None:
            # print("[ERROR] No previous homography available. Cannot rectify.")
            # Cannot proceed without a valid transform, return original or None
            # Returning None might be safer to signal failure downstream
            return None, None # Indicate failure
        H = last_successful_homography # Use fallback

    rectified = cv2.warpPerspective(frame, H, (target_w, target_h))
    return rectified, H # Return both frame and matrix used

# --- Other Functions (Unchanged) ---
def split_cells(frame):
    h, w = frame.shape[:2]
    cell_h = h // GRID_ROWS
    cell_w = w // GRID_COLS
    cells = []
    # Store as (y1, y2, x1, x2) for easier slicing
    for i in range(GRID_ROWS):
        for j in range(GRID_COLS):
            y1 = i * cell_h
            x1 = j * cell_w
            y2 = y1 + cell_h
            x2 = x1 + cell_w
            cells.append((y1, y2, x1, x2)) # row-major order: 0..8
    return cells

def detect_drop_start(video_path, threshold=500000, check_frames=150):
    """Checks the first few seconds for significant motion indicating drop."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video file: {video_path}")
        return 0
    prev_gray = None
    frame_index = 0
    max_check = check_frames # Limit check duration

    while frame_index < max_check:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0) # Larger blur for gross motion detection

        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            motion_score = np.sum(diff)
            # print(f"Frame {frame_index}, Motion score: {motion_score}") # Debug motion score
            if motion_score > threshold:
                print(f"[INFO] Significant motion detected around frame {frame_index}. Assuming drop ended.")
                cap.release()
                # Return a frame *after* the motion spike settles
                return min(frame_index + 5, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) -1) )

        prev_gray = gray
        frame_index += 1

    print("[INFO] No significant initial motion detected, starting from beginning.")
    cap.release()
    return 0 # Start from the beginning if no drop detected


def harris_score(gray_roi):
    """Calculates a score based on Harris corners for motion detection."""
    if gray_roi is None or gray_roi.size == 0: return 0
    # Parameters for Harris: blockSize=2, ksize=3, k=0.04
    dst = cv2.cornerHarris(np.float32(gray_roi), 2, 3, 0.04)
    # Simple score: sum of positive responses (more corners = higher score)
    score = np.sum(dst > 0.01 * dst.max())
    return score

# --- Main Execution Logic ---
def main():
    global last_successful_homography # Allow modification of the global fallback

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video: {VIDEO_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("[WARN] Could not get FPS, assuming 30.")
        fps = 30 # Provide a default if FPS is not available

    total_frames_to_process = int(fps * SECONDS)
    print(f"[INFO] Video FPS: {fps:.2f}, Processing {SECONDS} seconds ({total_frames_to_process} frames).")

    # Find the frame to start processing after initial drop/settling
    drop_end_frame = detect_drop_start(VIDEO_PATH, threshold=300000, check_frames=int(fps*5)) # Check first 5 secs
    print(f"[INFO] Starting processing from frame index: {drop_end_frame}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, drop_end_frame)

    # Initialize storage
    motion_matrix = np.zeros((SECONDS, 9), dtype=int) # Stores final 0/1 motion per second per cell
    frame_motion_buffer = [[] for _ in range(9)] # Stores 0/1 motion detection per frame within a second
    prev_harris_scores = [None] * 9 # Stores the last Harris score for each cell

    processed_frame_count = 0
    current_second = 0

    # Setup display window
    cv2.namedWindow("Robot Motion Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Robot Motion Detection", 720, 720) # Adjust size as needed

    # Read the very first frame to potentially initialize homography
    ret, initial_frame = cap.read()
    if ret:
        _, initial_H = rectify_frame(initial_frame) # Try to get an initial homography
        if initial_H is not None:
             last_successful_homography = initial_H
             print("[INFO] Initial homography calculated successfully.")
        else:
             print("[WARN] Could not calculate initial homography. Will retry on next frames.")
        # Reset video position after reading the initial frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, drop_end_frame)
    else:
        print("[ERROR] Could not read the first frame for initial homography.")
        cap.release()
        return


    # Main processing loop
    while processed_frame_count < total_frames_to_process:
        ret, frame = cap.read()
        if not ret:
            print(f"[WARN] Could not read frame at index {drop_end_frame + processed_frame_count}. End of video?")
            break

        # --- Step 1: Rectify Frame ---
        rectified_frame, _ = rectify_frame(frame) # H matrix is handled internally now

        if rectified_frame is None:
            print(f"[WARN] Skipping frame {drop_end_frame + processed_frame_count} due to rectification failure.")
            processed_frame_count += 1
            continue # Skip processing this frame

        # --- Step 2: Process Rectified Frame ---
        gray_rectified = cv2.cvtColor(rectified_frame, cv2.COLOR_BGR2GRAY)
        cells = split_cells(rectified_frame) # Calculate cell coordinates (should be constant)

        display_frame = rectified_frame.copy() # Create a copy for drawing overlays

        # --- Step 3: Detect Motion in Each Cell ---
        current_frame_has_motion = [0] * 9 # Track motion detected *in this specific frame*
        for i, (y1, y2, x1, x2) in enumerate(cells):
            roi_gray = gray_rectified[y1:y2, x1:x2]
            current_score = harris_score(roi_gray)

            motion_detected_in_cell = 0
            if prev_harris_scores[i] is not None:
                diff = abs(current_score - prev_harris_scores[i])
                if diff > HARRIS_DIFF_THRESHOLD:
                    motion_detected_in_cell = 1
                    current_frame_has_motion[i] = 1 # Mark motion for this frame

            frame_motion_buffer[i].append(motion_detected_in_cell)
            prev_harris_scores[i] = current_score # Update score for next comparison

            # --- Step 4: Draw Overlays (on the display copy) ---
            color = (0, 255, 0) if motion_detected_in_cell == 1 else (0, 0, 255)
            thickness = 2 if motion_detected_in_cell == 1 else 1
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)
            # Display the target Robot number (adjust font size/position if needed)
            cv2.putText(display_frame, str(display_index_map[i]), (x1 + 5, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


        # --- Step 5: Aggregate Results at Second Boundaries ---
        processed_frame_count += 1
        # Check if a second has passed
        if processed_frame_count > 0 and processed_frame_count % int(fps) == 0:
            sec_index = current_second # Index for the motion_matrix (0-based)
            if sec_index < SECONDS: # Ensure we don't write past the allocated array size
                print(f"[INFO] Processing end of second {sec_index + 1}") # 1-based for display
                for i in range(9):
                    if len(frame_motion_buffer[i]) > 0: # Avoid division by zero if second had 0 frames
                        ratio = sum(frame_motion_buffer[i]) / len(frame_motion_buffer[i])
                        if ratio >= MOTION_RATIO:
                            motion_matrix[sec_index][i] = 1
                        else:
                            motion_matrix[sec_index][i] = 0
                    else:
                         motion_matrix[sec_index][i] = 0 # No frames processed for this cell in this second

            # Reset buffer for the next second
            frame_motion_buffer = [[] for _ in range(9)]
            current_second += 1 # Move to the next second index


        # --- Step 6: Display Frame ---
        # Add second counter to display
        sec_display = int(processed_frame_count / fps) + 1
        cv2.putText(display_frame, f"Sec: {sec_display}", (10, display_frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow("Robot Motion Detection", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Quit key pressed. Exiting.")
            break

    # --- End of Loop ---
    cap.release()
    cv2.destroyAllWindows()

    # --- Step 7: Format and Write Output TXT ---
    output_motion_matrix_formatted = np.zeros((SECONDS, 9), dtype=int)
    for internal_idx in range(9): # internal index 0-8
        txt_col_idx = txt_order_mapping[internal_idx] - 1 # Convert Robot# (1-9) to 0-based column index
        if 0 <= txt_col_idx < 9:
            output_motion_matrix_formatted[:, txt_col_idx] = motion_matrix[:, internal_idx]
        else:
            print(f"[WARN] Invalid mapping for internal index {internal_idx} to Robot # {txt_order_mapping[internal_idx]}")

    print(f"[INFO] Writing motion data to: {OUTPUT_TXT}")
    try:
        with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
            # Header line - Adjust spacing if needed
            header = "Saniye\t" + "\t".join([f"Robot-{i+1}" for i in range(9)])
            f.write(header + "\n")
            # Data lines
            for i in range(SECONDS): # Iterate through seconds 0 to SECONDS-1
                # Format each number with potential padding (e.g., '{:>4}'.format(num)) if alignment is needed
                row_values = "\t".join([f"{output_motion_matrix_formatted[i][j]}" for j in range(9)])
                # Format second number (1-based)
                f.write(f"{i+1:<3}\t{row_values}\n") # Left align second number, use tabs for data
        print("[INFO] Output file written successfully.")
    except IOError as e:
        print(f"[ERROR] Could not write output file {OUTPUT_TXT}: {e}")

if __name__ == "__main__":
    main()