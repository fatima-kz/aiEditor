
# Cognitive Video AI

## Overview

Cognitive Video AI redefines video editing, making it as simple as typing a sentence. Powered by advanced AI, this tool automates video editing tasks based on user prompts, saving time and effort while enabling unmatched creativity. Whether you're a beginner or a professional, Cognitive Video AI bridges the gap between vision and execution.

## Features

1. **AI-Powered Editing**:
   - Automate edits using natural language prompts.
   - Perform tasks like trimming, adjusting brightness, enhancing contrast, sharpening, slowing or speeding up videos, and removing objects.

2. **User-Centric Design**:
   - Designed for non-experts and professionals alike.
   - Simplifies complex video editing workflows.

3. **Seamless Integration**:
   - Leverages advanced AI tools (Gemini 2.0, YOLOv8) for precise and intelligent edits.
   - Outputs high-quality videos with minimal user intervention.

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- `pip` package manager
- [Streamlit](https://streamlit.io) for UI
- Required libraries:
  - `opencv-python`
  - `numpy`
  - `ultralytics`
  - `streamlit`
  - `google-generativeai`
  - `python-dotenv`


### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Set Up Environment Variables

Create a `.env` file in the root directory and add your API keys:

```env
GEMINI=<Your Gemini API Key>
```

### Step 3: Run the Application

Start the Streamlit app:

```bash
streamlit run main4.py
```

## Usage Instructions

1. Upload your video via the sidebar in the Streamlit app.
2. Enter your natural language prompt in the text box (e.g., "Enhance brightness when a person is detected").
3. Click **Generate Content** to process the video using AI-powered tools.
4. Download and preview the edited video.



### Supported Functions

- **Adjust Brightness**: Modify the brightness within a specified time range.
- **Enhance Contrast**: Apply histogram equalization for better contrast.
- **Sharpen Video**: Increase frame sharpness using convolution filters.
- **Trim Video**: Remove specific time ranges.
- **Slow Down/Speed Up Video**: Adjust playback speed for selected segments.
- **Remove Objects**: Detect and remove objects using YOLOv8.

## Code Structure

- **`video_editor.py`**: Core video processing functions.
- **`tool_executor.py`**: Maps AI-generated function calls to video processing methods.
- **`main4.py`**: Streamlit-based user interface for interacting with the system.

## Example Prompts

- "Crop the parts of the video where you see birds."
- "Remove the potted plant from the background."
- "Enhance the brightness only when a person is detected."


## Contributing

We welcome contributions! Please fork the repository and submit a pull request with your changes.



## Acknowledgements

- The **YOLOv8** model for object detection.
- OpenAI's **Gemini API** for natural language understanding.
- **Streamlit** for the interactive user interface.

