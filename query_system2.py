# query_system.py
import openai
import json

class OpenAIIntegration:
    def __init__(self, api_key):
        openai.api_key = api_key

    def query_llm(self, context, user_query, system_prompt=None):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(context)},
            {"role": "user", "content": user_query}
        ]

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",  # Specify the OpenAI model
                messages=messages,
                functions=self.get_tools()
            )
            return response.to_dict()
        except openai.error.OpenAIError as e:
            print(f"Error querying OpenAI: {e}")
            return None

    def get_tools(self):
        return [
            {
                "name": "adjust_brightness",
                "description": "Increase or decrease the brightness of a video segment.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start_time": {"type": "number", "description": "Start time in seconds."},
                        "end_time": {"type": "number", "description": "End time in seconds."},
                        "value": {"type": "number", "description": "Brightness level."}
                    },
                    "required": ["start_time", "end_time", "value"]
                }
            },
            {
                "name": "adjust_contrast",
                "description": "Enhance or reduce the contrast of a video segment.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start_time": {"type": "number", "description": "Start time in seconds."},
                        "end_time": {"type": "number", "description": "End time in seconds."},
                        "level": {"type": "number", "description": "Contrast level."}
                    },
                    "required": ["start_time", "end_time", "level"]
                }
            },
        ]
