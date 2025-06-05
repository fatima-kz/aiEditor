import google.generativeai as genai
import dotenv
import os
import time
from typing import Callable, Dict
from video_editor import adjust_brightness, enhance_contrast, sharpen_video, trim_video, slow_down_video, speed_up_video, remove_object
from tool_executor import execute_tool_calls


dotenv.load_dotenv()
tools = [adjust_brightness, enhance_contrast, sharpen_video, trim_video, slow_down_video, speed_up_video, remove_object]

gemini_api_key = os.getenv("GEMINI")
genai.configure(api_key=gemini_api_key)

video_file_name = "sVid.mp4"

print(f"Uploading file...")
video_file = genai.upload_file(path=video_file_name)
print(f"Completed upload: {video_file.uri}")

while video_file.state.name == "PROCESSING":
    print('.', end='')
    time.sleep(10)
    video_file = genai.get_file(video_file.name)

if video_file.state.name == "FAILED":
    raise ValueError("File processing failed.")

print(f"File ready: {video_file.name}")

prompt = "Trim only the parts of the video where a person is not being detected. The person appears twice in the video."

model = genai.GenerativeModel(model_name="gemini-2.0-flash-exp", tools=tools, system_instruction=f"You will always give the initial video input path {video_file_name}. Always give the same output path for the edited video with the name \'output.mp4\'. For slowing and speeding video in sequence, adjust the timestamps acc to the factor you give. When trimming the video twice, use the start and end times of the second trim based on the total duration of the latest edited video.")

print("Making LLM inference request...")
response = model.generate_content([video_file, prompt], request_options={"timeout": 600})

if response:
    print("Response received:")
    print(response)
else:
    print("No response received.")
    
for part in response.parts:
    if fn := part.function_call:
        args = ", ".join(f"{key}={val}" for key, val in fn.args.items())
        print(f"{fn.name}({args})")
        
execute_tool_calls(response)

