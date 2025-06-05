import cv2
import numpy as np
from typing import Callable, Dict
from video_editor import (
    adjust_brightness, enhance_contrast,
    sharpen_video, trim_video, slow_down_video, speed_up_video, remove_object
)

FUNCTION_MAPPING: Dict[str, Callable] = {
    "adjust_brightness": adjust_brightness,
    "enhance_contrast": enhance_contrast,
    "sharpen_video": sharpen_video,
    "trim_video": trim_video,
    "slow_down_video": slow_down_video,
    "speed_up_video": speed_up_video,
    "remove_object": remove_object
}

def execute_tool_calls(response):
    """
    Parses Gemini tool calls, maps them to the corresponding functions, and executes them.

    Args:
        response: The Gemini response containing tool calls.
    """

    for part in response.parts:
        if fn := part.function_call:
            function_name = fn.name
            args = fn.args

            if function_name in FUNCTION_MAPPING:
                func = FUNCTION_MAPPING[function_name]

                # video_input_path = args.get("video_path")
                # video_output_path = args.get("output_path")

                # if not video_input_path or not video_output_path:
                #     print(f"Error: Missing video paths for {function_name}. Skipping this call.")
                #     continue

                function_args = {
                    key: float(val) if isinstance(val, str) and val.replace('.', '', 1).isdigit() else val
                    for key, val in args.items()
                    if key not in ["video_input_path", "video_output_path"]
                }

                function_args["video_path"] = args.get("video_path")
                function_args["output_path"] = args.get("output_path")

                print(f"Executing: {function_name} with args {function_args}")
                func(**function_args)
            else:
                print(f"Warning: Function '{function_name}' is not implemented.")

    print("All tool calls executed successfully.")