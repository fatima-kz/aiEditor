
import requests
import json

MODEL = "llama3.2"  # Specify the model you are using

class OllamaIntegration:
    def __init__(self, api_url, api_key):
        self.api_url = api_url
        self.api_key = api_key

    def query_llm(self, context, user_query, system_prompt=None):
        # Combine context and user query into a single prompt
        # prompt = self._generate_prompt(context, user_query)
        
        tools = [
    {
        "type": "function",
        "function": {
            "name": "adjust_brightness",
            "description": "Increase or decrease the brightness of a video segment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {
                        "type": "number",
                        "description": "The start time in seconds for the brightness adjustment.",
                    },
                    "end_time": {
                        "type": "number",
                        "description": "The end time in seconds for the brightness adjustment.",
                    },
                    "value": {
                        "type": "number",
                        "description": "The brightness level to apply. Positive values increase brightness, negative values decrease it.",
                    },
                },
                "required": ["start_time", "end_time", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sharpen_video",
            "description": "Apply sharpening to a video segment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {
                        "type": "number",
                        "description": "The start time in seconds for the sharpening effect.",
                    },
                    "end_time": {
                        "type": "number",
                        "description": "The end time in seconds for the sharpening effect.",
                    },
                    "intensity": {
                        "type": "number",
                        "description": "The intensity of the sharpening effect (e.g., 1.0 for normal, higher for stronger).",
                    },
                },
                "required": ["start_time", "end_time", "intensity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "adjust_contrast",
            "description": "Enhance or reduce the contrast of a video segment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {
                        "type": "number",
                        "description": "The start time in seconds for the contrast adjustment.",
                    },
                    "end_time": {
                        "type": "number",
                        "description": "The end time in seconds for the contrast adjustment.",
                    },
                    "level": {
                        "type": "number",
                        "description": "The contrast level to apply. Positive values increase contrast, negative values decrease it.",
                    },
                },
                "required": ["start_time", "end_time", "level"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "smooth_video",
            "description": "Apply smoothing (Gaussian blur) to a video segment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {
                        "type": "number",
                        "description": "The start time in seconds for the smoothing effect.",
                    },
                    "end_time": {
                        "type": "number",
                        "description": "The end time in seconds for the smoothing effect.",
                    },
                    "kernel_size": {
                        "type": "number",
                        "description": "The kernel size for the Gaussian blur. Larger values produce stronger smoothing.",
                    },
                },
                "required": ["start_time", "end_time", "kernel_size"],
            },
        },
    },
]

        
        prompt = f"""
        
        Context:{context}\n\n
        User Query: {user_query}
        """
        print("Context: " + str(context))

        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": json.dumps(context)
            },
            {
                "role": "user",
                "content": user_query
            }
        ]
        
        payload = {        
            "model": MODEL,
            "messages": messages,
            "stream": False,
            "tools": tools
        }
        
        
        # payload = {
        #     "model": MODEL,  # Specify the model you are using
        #     "prompt": prompt,
        #     "tools": tools,
        # }
        headers = {"Content-Type": "application/json"}

        # payload = json.dumps(payload)
        
        try:
            # print("Payload: " + str(payload))
            print("Passing payload to LLM...")
            
            
            
            response = requests.post(self.api_url, headers=headers, json=payload)
            
            #print("Response from LLM:" + response.text)
            # print("Response tool call: " + response.tool_call)
            
            response.raise_for_status()  # Raise exception for HTTP errors
            return response.json()  # Return the LLM response
        except requests.exceptions.RequestException as e:
            print(f"Error querying LLM: {e}")
            return None

    def _generate_prompt(self, context, user_query):
        # Convert context to a well-structured prompt string
        transcription = context.get("transcription", "No transcription available.")
        object_detection = context.get("object_detection", [])
        objects_summary = "\n".join([
            f"Frame {item['frame']} (Timestamp: {item['timestamp']}) {', '.join(item['detected_objects'])}  {', '.join(str(item['bounding_boxes'])) } {', '.join(str(item['confidences'])) }"
            for item in object_detection
        ])
        return f"""
        Context:
        Transcription: {transcription}
        Detected objects:
        {objects_summary}

        User Query: {user_query}

        Respond with the required actions and tools in JSON format.
        """
