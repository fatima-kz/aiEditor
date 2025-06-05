import cv2
import os
from tool_executor import ToolExecutor

def edit_video(video_path, edits, output_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Apply edits based on timestamps
        for edit in edits:
            if int(edit['start']) <= frame_idx / cap.get(cv2.CAP_PROP_FPS) <= int(edit['end']):
                frame = ToolExecutor.call_tool(frame, edit['tool'], **edit['args'])

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Edited video saved to {output_path}")
