from frame_extractor import extract_frames_with_timestamps
from audio_transcriber import transcribe_audio
from object_detector import detect_objects
from query_system import OllamaIntegration
from video_orchestrator import edit_video
import torch
import os
import pandas as pd
from ultralytics import YOLO
import json

# Load the model



# Paths
video_path = "fatima.mp4"
output_dir = "frames"
edited_video_path = "edited_video.mp4"

# Step 1: Extract frames
frames_data = extract_frames_with_timestamps(video_path, output_dir, frame_rate=5)
frames_data.to_csv(os.path.join(output_dir, "frames_timestamps.csv"), index=False)

# Step 2: Transcribe audio
transcription = transcribe_audio(video_path)

# Step 3: Object detection
model = YOLO("yolov5s.pt")  # Replace with the path to the YOLOv5s model if downloaded manually
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

results_data = []
for _, row in frames_data.iterrows():
    frame_path = row['frame']
    timestamp = row['timestamp']
    detection_results = detect_objects(frame_path, model)

    results_data.append({
        "frame": frame_path,
        "timestamp": timestamp,
        "detected_objects": detection_results["class_names"],
        "bounding_boxes": detection_results["boxes"].tolist(),
        "confidences": detection_results["confidences"].tolist()
    })
    # print(f"Results for {frame_path}: {detection_results['class_names']}")

# print("Results Data: " + str(results_data))

results_df = pd.DataFrame(results_data)
results_df.to_csv(os.path.join(output_dir, "detection_results.csv"), index=False)

# Step 4: Query LLM and edit video
ollama_api = OllamaIntegration(api_url="http://localhost:11434/api/chat", api_key="your_api_key")
context = {"transcription": transcription, "object_detection": results_df.to_string()}


user_query = "Enhnance the contrast of the part of the video in which a person is detected in the frames."

system_prompt = """
    You are an advanced video editing assistant.
    
    You are supposed to look at the context we give, which contains information for each frame like the objects detected and their bounding box coordinates as well as the time stamp of each frame, now my tools actually work based on the timestamp of the video, each tool can be applied to a certain part of the video based on starting and ending time, there may be additional values for some tools but the time stamps are a must. I want the llm to understand the context and check where, in time, each object is in the video as well as how its moving, after that the llm should call tools and provide arguments based on user query with respect to the video

    Your task is to analyze the video context provided, which contains the following information:
    - A list of detected objects in each frame, their bounding box coordinates, and the timestamp for each frame.

    Based on the user query and the video context:
    1. **Understand the Video Context:**
    - Identify which objects are present in the video and their locations over time.
    - Analyze the movement of objects across frames to infer trajectories and interactions.
    - Map objects and events in the video to their corresponding timestamps.

    2. **Determine Required Tools:**
    - Based on the user query, identify which editing tools are needed to achieve the desired modifications.
    - Each tool must be applied to specific parts of the video, determined by their start and end timestamps.

    3. **Generate Tool Calls:**
    - For each required tool, provide a structured tool call with accurate arguments.
    - **Mandatory arguments** for all tools:
        - `start_time`: The starting time of the video segment where the tool should be applied.
        - `end_time`: The ending time of the video segment where the tool should be applied.
    - Additional arguments (specific to each tool) must be provided as required by the user query and the video context.

    4. **Parameter Values:**
    - Ensure all parameter values are logical and aligned with the user query.
    - Use the movement and behavior of objects over time to determine relevant timestamps and parameter values.
    - Provide realistic values for all additional arguments (e.g., intensity, value) based on the context and query.

    ### Example Tool Call Format
    For calling tools, use the following structure:
    {
        "function": {
            "name": "RequiredFunctionName",
            "arguments": {
                "start_time": <value>,
                "end_time": <value>,
                "intensity": <value>
            }
        }
    }
    ### Guidelines
    - Always prioritize timestamps (`start_time` and `end_time`) based on object presence or movement in the video.
    - Additional parameters should only be included if relevant to the tool and user query.
    - Ensure tool calls reflect the user's intent and are directly applicable to the provided video context.

    ### Example Scenarios
    1. **User Query: "Sharpen the area where the car moves from 1s to 3s."**
    - Identify the timestamps where the car is present.
    - Generate a tool call for the `sharpen_video` tool with the appropriate timestamps and intensity.

    **Example Output:**
    {
        "function": {
            "name": "sharrpen_video",
            "arguments": {
                "start_time": 1,
                "end_time": 3,
                "intensity": 1.5
            }
        }
    }
    2. **User Query: "Brighten the section where the traffic light is visible."**
    - Locate the timestamps where the traffic light is detected.
    - Generate a tool call for the `adjust_brightness` tool with the relevant timestamps and brightness value.

    **Example Output:**
    {
        "function": {
            "name": "adjust_brightness",
            "arguments": {
                "start_time": 0,
                "end_time": 2,
                "intensity": 1.5
            }
        }
    }
    Your goal is to provide accurate tool calls that align with the user query and video context, ensuring the output is both practical and precise.
    """


llm_response = ollama_api.query_llm(context, user_query, system_prompt)
formatted_response = json.dumps(llm_response, indent=4)
print("Formatted Response:")
print(formatted_response)

# Parse and pretty-print the content
content = llm_response["message"]["content"]

# Split JSON objects if needed
content_parts = content.split("\n\n")

formatted_parts = []
for part in content_parts:
    try:
        parsed_json = json.loads(part)
        formatted_parts.append(json.dumps(parsed_json, indent=4))
    except json.JSONDecodeError:
        # Handle plain text or poorly formatted JSON
        formatted_parts.append(part)

# Combine and print the formatted content
formatted_content = "\n\n".join(formatted_parts)
print("Formatted Content:")
print(formatted_content)

# if llm_response:
    # edits = [{"tool": action["tool"], "start": action["parameters"].get("start_time", 0),
    #           "end": action["parameters"].get("end_time", 0), "args": [action["parameters"].get("value", 0)]}
    #          for action in llm_response]
    # Process the LLM response to extract tool calls and construct the edits list

    # Check if tool_calls are present in the response

edits = []
if llm_response:
    print('bala')
    if "tool_calls" in llm_response["message"]:
        print('bala1')
        for action in llm_response["message"]["tool_calls"]:
            tool_name = action.get("function", {}).get("name")
            parameters = action.get("function", {}).get("arguments", {})
            
            print("Tool Name:", tool_name)
            print("Parameters:", parameters)

            # Build the edit dictionary for each tool call
            edit = {
                "tool": tool_name,
                "start": parameters.get("start_time", 0),
                "end": parameters.get("end_time", 0),
                "args": {key: value for key, value in parameters.items() if key not in ["start_time", "end_time"]},
            }
            edits.append(edit)
            
        print("Edits: " + str(edits))

        edit_video(video_path, edits, edited_video_path)
