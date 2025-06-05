import pandas as pd
import os
import cv2

def extract_frames_with_timestamps(video_path, output_folder, frame_rate=1):
    """
    Extract frames from a video and save timestamps.
    
    Args:
        video_path (str): Path to the input video file.
        output_folder (str): Directory to save the extracted frames.
        frame_rate (int): Extract 1 frame per frame_rate frames.
        
    Returns:
        pd.DataFrame: DataFrame containing frame filenames and timestamps.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print("Error: Cannot open video file.")
        return None

    # Get FPS and total frames
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video FPS: {fps}, Total Frames: {total_frames}")

    frame_count = 0
    saved_count = 0
    data = []

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        if frame_count % frame_rate == 0:
            timestamp = frame_count / fps  # Calculate timestamp
            frame_filename = os.path.join(output_folder, f"frame{saved_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            data.append({"frame": frame_filename, "timestamp": timestamp})
            print(f"Saved: {frame_filename}, Timestamp: {timestamp:.2f}s")
            saved_count += 1

        frame_count += 1

    video_capture.release()
    print(f"Extraction complete. {saved_count} frames saved to '{output_folder}'.")

    # Return a DataFrame with frame filenames and timestamps
    return pd.DataFrame(data)
