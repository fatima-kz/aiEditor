from frame_extractor import extract_frames_with_timestamps
from audio_transcriber import transcribe_audio
from object_detector import detect_objects
from video_orchestrator import edit_video
import torch
import os
import pandas as pd
from ultralytics import YOLO
import json
import openai
from openai import OpenAI
import dotenv


#Load env
dotenv.load_dotenv()
openai_api_key = os.getenv("OPENAI")
gemini_api_key = os.getenv("GEMINI")

# Load the model
client = OpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Paths
video_path = "1734021617.mp4"
output_dir = "frames"
edited_video_path = "edited_video.mp4"

# Step 1: Extract frames
frames_data = extract_frames_with_timestamps(video_path, output_dir, frame_rate=20)
frames_data.to_csv(os.path.join(output_dir, "frames_timestamps.csv"), index=False)

# Step 2: Transcribe audio
transcription = transcribe_audio(video_path)

# Step 3: Object detection
model = YOLO("yolov5s.pt")  # Replace with the path to the YOLOv5s model if downloaded manually

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

results_df = pd.DataFrame(results_data)
results_df.to_csv(os.path.join(output_dir, "detection_results.csv"), index=False)

# Step 4: Query OpenAI and edit video
class OpenAIIntegration:
    def __init__(self, api_key):
        openai.api_key = api_key

        

    def query_llm(self, messages, functions=None):
        

        try:
            response = client.chat.completions.create(
                model="gemini-2.0-flash-exp",  # Specify the model
                messages=messages,
                tools=functions
            )
            return response
        except Exception as e:
            print(f"Error querying OpenAI: {e}")
            return None

functions = [
  {
      "type": "function",
      "function": {
          "name": "adjust_brightness",
          "description": "Adjust the brightness of a video segment.",
          "parameters": {
              "type": "object",
              "properties": {
                  "start_time": {"type": "number", "description": "Start time in seconds."},
                  "end_time": {"type": "number", "description": "End time in seconds."},
                  "value": {"type": "number", "description": "Brightness adjustment value."}
              },
              "required": ["start_time", "end_time", "value"]
          }
      }
  },
  {
      "type": "function",
      "function": {
          "name": "sharpen_video",
          "description": "Sharpen a video segment.",
          "parameters": {
              "type": "object",
              "properties": {
                  "start_time": {"type": "number", "description": "Start time in seconds."},
                  "end_time": {"type": "number", "description": "End time in seconds."},
                  "intensity": {"type": "number", "description": "Sharpening intensity."}
              },
              "required": ["start_time", "end_time", "intensity"]
          }
      }
  },
  {
      "type": "function",
      "function": {
          "name": "adjust_contrast",
          "description": "Adjust the contrast of a video segment.",
          "parameters": {
              "type": "object",
              "properties": {
                  "start_time": {"type": "number", "description": "Start time in seconds."},
                  "end_time": {"type": "number", "description": "End time in seconds."},
                  "level": {"type": "number", "description": "Contrast adjustment level."}
              },
              "required": ["start_time", "end_time", "level"]
          }
      }
  },
  {
  "type": "function",
  "function": {
    "name": "hide_object",
    "description": "Makes an object invisible in an image by filling or blurring its region.",
    "parameters": {
      "type": "object",
      "properties": {
        "start_time": {"type": "number", "description": "Start time in seconds."},
        "end_time": {"type": "number", "description": "End time in seconds."},
        "coordinates": {
          "type": "array",
          "items": {"type": "number"},
          "description": "Bounding box of the object in the format [x_min, y_min, x_max, y_max]."
        }
      },
      "required": ["start_time", "end_time", "coordinates"]
    }
  }
}


]


context = {"transcription": transcription, "object_detection": results_df.to_dict()}
user_query2 = "Enhance the contrast of the part of the video in which a person is detected in the frames plus Understand what is going on in the video and give me a short summary of it. Does the video contain a cat?"
#user_query2 = "Understand what is going on in the video and give me a short summary of it. Does the video contain a cat?"
user_query = "hide the person from the video"
messages = [
            {"role": "system", "content": "You are an advanced video editing assistant who gives tool calls"},
            {"role": "user", "content": "Analyze the following video context and respond to the query."},
            {"role": "user", "content": json.dumps(context)},
            {"role": "user", "content": user_query}
        ]

messages2 = [
            {"role": "system", "content": "you are an advanced video summarizer. GIVE DESCRIPTION OF VIDEO CONTEXT HERE and NO TOOL CALLS"},
            {"role": "user", "content": "Analyze the following video context and respond to the query."},
            {"role": "user", "content": json.dumps(context)},
            {"role": "user", "content": user_query}
        ]

api_key = openai_api_key
openai_api = OpenAIIntegration(api_key)
response = openai_api.query_llm(messages, functions)
response2 = openai_api.query_llm(messages2)


#print("Response: ",response)
print("Response Description: ",response2.choices[0].message)
print("Response messege: ",response.choices[0].message)


if response:
    # Access the tool_calls attribute instead of function_call
    tool_calls = response.choices[0].message.tool_calls
    edits = []

    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        parameters = json.loads(tool_call.function.arguments)  # Convert arguments from string to dictionary
        edits.append({
            "tool": tool_name,
            "start": parameters.get("start_time"),
            "end": parameters.get("end_time"),
            "args": {key: value for key, value in parameters.items() if key not in ["start_time", "end_time"]}
        })

    # Perform video editing
    edit_video(video_path, edits, edited_video_path)
