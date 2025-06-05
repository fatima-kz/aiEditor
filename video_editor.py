import cv2
import numpy as np
import os
from ultralytics import YOLO


def adjust_brightness(video_path: str, output_path: str, start_time: float, end_time: float, value: float):
    """
    Adjust the brightness of a video within a given timestamp range.

    Args:
        video_path: Path to the input video file.
        output_path: Path to save the edited video.
        start_time: Start time in seconds to apply the adjustment.
        end_time: End time in seconds to stop applying the adjustment.
        value: Brightness adjustment value. the value should range from -100.0 to 100.0.

    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, 
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if start_frame <= frame_count <= end_frame:
            frame = cv2.convertScaleAbs(frame, alpha=1, beta=float(value))  # Adjust brightness
        
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    print("Brightness adjustment complete.")



def enhance_contrast(video_path: str, output_path: str, start_time: float, end_time: float):
    """
    Enhance the contrast of a video using histogram equalization in a given timestamp range.

    Args:
        video_path: Path to the input video file.
        output_path: Path to save the edited video.
        start_time: Start time in seconds.
        end_time: End time in seconds.

    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps,
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if start_frame <= frame_count <= end_frame:
            # Convert to YUV color space
            yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            # Equalize the histogram of the luminance channel (Y channel)
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
            # Convert back to BGR color space
            frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    print("Contrast enhancement complete.")



def sharpen_video(video_path: str, output_path: str, start_time: float, end_time: float):
    """
    Sharpen the video frames in a specific timestamp range.

    Args:
        video_path: Path to the input video file.
        output_path: Path to save the edited video.
        start_time: Start time in seconds.
        end_time: End time in seconds.

    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps,
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if start_frame <= frame_count <= end_frame:
            frame = cv2.filter2D(frame, -1, kernel)  # Apply sharpening filter
        
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    print("Sharpening complete.")


import cv2
import os

def trim_video(video_path: str, output_path: str, start_time_list: list[float], end_time_list: list[float]):
    """
    Remove multiple timestamp ranges from the video.

    Args:
        video_path: Path to the input video file.
        output_path: Path to save the modified video.
        start_time_list: List of start times (in seconds) specifying the beginning of each range to remove.
        end_time_list: List of end times (in seconds) specifying the end of each range to remove.

    NOTE: Ensure time ranges are in seconds, do not overlap, and are of equal length.
    """
    import cv2
    import os

    # Ensure the lengths of start_time_list and end_time_list match
    if len(start_time_list) != len(end_time_list):
        raise ValueError("The length of start_time_list and end_time_list must match.")

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Convert time ranges to frame ranges
    frame_ranges = [
        (int(start * fps), int(end * fps)) for start, end in zip(start_time_list, end_time_list)
    ]

    # Validate time ranges
    for start_frame, end_frame in frame_ranges:
        if start_frame < 0 or end_frame >= total_frames or start_frame >= end_frame:
            raise ValueError(f"Invalid range: ({start_frame}, {end_frame})")

    # Sort ranges by start frame
    frame_ranges.sort()

    # Setup the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_output_path = "temp_" + output_path
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (frame_width, frame_height))

    # Process the video
    frame_count = 0
    current_range_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Check if the current frame is within any of the ranges to remove
        if current_range_idx < len(frame_ranges):
            start_frame, end_frame = frame_ranges[current_range_idx]
            if start_frame <= frame_count <= end_frame:
                # Skip writing frames in this range
                frame_count += 1
                continue
            elif frame_count > end_frame:
                # Move to the next range
                current_range_idx += 1

        # Write frames outside the ranges
        out.write(frame)
        frame_count += 1

    # Release resources
    cap.release()
    out.release()

    # Rename the temp file to the final output path
    if os.path.exists(output_path):
        os.remove(output_path)
    os.rename(temp_output_path, output_path)

    print(f"Video parts removed. Output saved to {output_path}")
    return True




def slow_down_video(video_path: str, output_path: str, factor: float, start_time: float, end_time: float):
    """
    Slow down the video playback within a specific timestamp range by a given factor.

    Args:
        video_path: Path to the input video file.
        output_path: Path to save the slowed-down video.
        factor: The factor by which to slow down the video. E.g., 2.0 = half speed.
        start_time: Start time in seconds to apply the slowdown effect.
        end_time: End time in seconds to stop the slowdown effect.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    output_path_tmp = "tmp_" + output_path

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path_tmp, fourcc, fps, (width, height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if start_frame <= frame_count <= end_frame:
            # Write each frame multiple times to slow down
            for _ in range(int(factor)):
                out.write(frame)
        else:
            # Write the frame normally outside the range
            out.write(frame)
        
        frame_count += 1

    cap.release()
    out.release()
    os.remove(output_path)
    os.rename(output_path_tmp, output_path)
    print(f"Video slowed down successfully: {output_path}")


def speed_up_video(video_path: str, output_path: str, factor: float, start_time: float, end_time: float):
    """
    Speed up the video playback within a specific timestamp range by a given factor.

    Args:
        video_path: Path to the input video file.
        output_path: Path to save the sped-up video.
        factor: The factor by which to speed up the video. E.g., 2.0 = double speed.
        start_time: Start time in seconds to apply the speed-up effect.
        end_time: End time in seconds to stop the speed-up effect.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    output_path_tmp = "tmp_" + output_path

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path_tmp, fourcc, fps, (width, height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if start_frame <= frame_count <= end_frame:
            # Skip frames to speed up
            if frame_count % int(factor) == 0:
                out.write(frame)
        else:
            # Write the frame normally outside the range
            out.write(frame)
        
        frame_count += 1

    cap.release()
    out.release()
    # Release the temporary video
    
    # cap1 = cv2.VideoCapture(output_path_tmp)
    # if not cap1.isOpened():
    #     print(f"Error: Cannot open video file {video_path}")
    #     return

    # fps1 = cap.get(cv2.CAP_PROP_FPS)
    # width1 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height1 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # fourcc1 = cv2.VideoWriter_fourcc(*'mp4v')
    # out1 = cv2.VideoWriter(output_path, fourcc1, fps1, (width1, height1))
    
    
    
    # cap1.release()
    # out1.release()
    
    os.remove(output_path)
    os.rename(output_path_tmp, output_path)
    
    print(f"Video sped up successfully: {output_path}")
    

def remove_object(video_path: str, output_path: str, object_name: str):
    """
    Detects and removes a specific object from a video using YOLOv8 and bounding box inpainting.

    Args:
        video_path: Path to the input video file.
        output_path: Path to save the edited video.
        object_name: Name of the object to detect and remove.
    """
    # Load the YOLOv8 model
    model = YOLO('yolov8n.pt')  # Use the YOLOv8 Nano model (small and fast)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file '{video_path}'.")
        return

    # Video writer setup
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Processing video '{video_path}' to remove object '{object_name}'...")

    # Process video frame by frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv8 detection
        results = model(frame)
        detections = results[0].boxes

        for det in detections:
            # Extract bounding box and class
            x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
            cls = det.cls[0]
            label = model.names[int(cls)]

            # Check if the detected object matches the specified object name
            if label == object_name:
                # Create a mask and inpaint the region
                mask = np.zeros(frame.shape[:2], np.uint8)
                mask[y1:y2, x1:x2] = 255
                frame = cv2.inpaint(frame, mask, inpaintRadius=10, flags=cv2.INPAINT_TELEA)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Object '{object_name}' removed and video saved at: {output_path}")
