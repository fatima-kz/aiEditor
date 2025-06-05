import streamlit as st
import google.generativeai as genai
import dotenv
import os
import time
from video_editor import adjust_brightness, enhance_contrast, sharpen_video, trim_video, slow_down_video, speed_up_video, remove_object
from tool_executor import execute_tool_calls
import cv2

dotenv.load_dotenv()
gemini_api_key = os.getenv("GEMINI")
genai.configure(api_key=gemini_api_key)

tools = [adjust_brightness, enhance_contrast, sharpen_video, trim_video, slow_down_video, speed_up_video, remove_object]

st.title("Cognitive Video Editor")
st.sidebar.header("Input Video Configuration")

video_file_name = st.sidebar.text_input("Enter relative path of input video:", "newVid.mp4")

if "video_uploaded" not in st.session_state:
    st.session_state.video_uploaded = False

if "response_generated" not in st.session_state:
    st.session_state.response_generated = False

if st.sidebar.button("Process Video"):
    try:
        if "video_file" in st.session_state and st.session_state.video_file.name == video_file_name:
            st.success("Video already uploaded.")
        else:
            st.write("Uploading file...")
            video_file = genai.upload_file(path=video_file_name)
            st.write(f"Completed upload: {video_file.uri}")

            with st.spinner("Processing video..."):
                while video_file.state.name == "PROCESSING":
                    time.sleep(10)
                    video_file = genai.get_file(video_file.name)

            if video_file.state.name == "FAILED":
                st.error("File processing failed.")
            else:
                st.success(f"File ready: {video_file.name}")
                st.session_state.video_uploaded = True
                st.session_state.video_file = video_file

    except Exception as e:
        st.error(f"An error occurred: {e}")

if "video_file" in st.session_state and st.session_state.video_uploaded:
    prompt = st.text_area("Enter your editing prompt:", "Enhance the brightness when you detect a person.")

    if st.button("Generate Content"):
        try:
            model = genai.GenerativeModel(
                model_name="gemini-2.0-flash-exp",
                tools=tools,
                system_instruction=f"You will always give the initial video input path {video_file_name}. Always give the same output path for the edited video with the name \'output.mp4\'. For slowing and speeding video in sequence, adjust the timestamps acc to the factor you give. When trimming the video twice, use the start and end times of the second trim based on the total duration of the latest edited video."
            )

            st.write("Making LLM inference request...")
            with st.spinner("Editing video... This may take a while."):
                response = model.generate_content([st.session_state.video_file, prompt], request_options={"timeout": 600})

            if response:
                st.success("Response received.")
                st.session_state.response_generated = True
                st.session_state.response = response

                for part in response.parts:
                    if fn := part.function_call:
                        args = ", ".join(f"{key}={val}" for key, val in fn.args.items())
                        st.write(f"{fn.name}({args})")

                with st.spinner("Executing tool calls..."):
                    execute_tool_calls(response)
                st.success("Tool execution completed.")

                output_video_path = "output.mp4"
                fixed_video_path = "output_fixed.mp4"

                if os.path.exists(output_video_path):
                    try:
                        cap = cv2.VideoCapture(output_video_path)
                        fourcc = cv2.VideoWriter_fourcc(*'X264')
                        out = cv2.VideoWriter(fixed_video_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                                            (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
                        
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                break
                            out.write(frame)

                        cap.release()
                        out.release()

                        if os.path.exists(fixed_video_path):
                            st.video(fixed_video_path)
                        else:
                            st.error(f"Converted video not found at {fixed_video_path}.")
                    except Exception as e:
                        st.error(f"Error occurred while converting video: {e}")
                else:
                    st.error(f"Original video not found at {output_video_path}.")
            else:
                st.error("No response received.")

        except Exception as e:
            st.error(f"An error occurred: {e}")
