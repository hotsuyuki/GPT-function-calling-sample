import dotenv
import huggingface_hub
import json
import openai
import os
from typing import Union


class FunctionCallingGPT:
    def __init__(self, model: str = "gpt-3.5-turbo", is_verbose: bool = False) -> None:
        self.model = model
        self.is_verbose = is_verbose

        dotenv.load_dotenv()
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

        openai.api_key = OPENAI_API_KEY
        self.huggingface_inference_client = huggingface_hub.InferenceClient(token=HUGGINGFACE_API_KEY)

        self.system_content = ""

        with open(os.path.join(os.getenv("PWD"), "functions.json"), "r") as f:
            self.functions = json.load(f)

        if self.is_verbose:
            print("OPENAI_API_KEY =", OPENAI_API_KEY[:8], "...")
            print("HUGGINGFACE_API_KEY =", HUGGINGFACE_API_KEY[:8], "...")
            print()
            print(f"{self.system_content = }")
            print()
            print(f"{self.functions = }")
            print()

    def __call__(self, user_content: str) -> str:
        messages = [
            {"role": "system", "content": self.system_content},
            {"role": "user", "content": user_content}
        ]
        response = None
        is_stop = False

        while not is_stop:
            response = openai.ChatCompletion.create(model=self.model, messages=messages, functions=self.functions)

            if self.is_verbose:
                print(f"{messages = }")
                print()
                print(f"{response = }")
                print()

            if response["choices"][0]["message"].get("function_call"):
                messages.append({
                    "role": "assistant",
                    "content": response["choices"][0]["message"]["content"],
                    "function_call": response["choices"][0]["message"]["function_call"]
                })

                function_name = response["choices"][0]["message"]["function_call"]["name"]
                function_arguments = response["choices"][0]["message"]["function_call"].get("arguments")
                if function_arguments and isinstance(function_arguments, str):
                    function_arguments = json.loads(function_arguments)
                function_content = self.call_huggingface_inference_api(function_name, function_arguments)

                messages.append({
                    "role": "function",
                    "name": response["choices"][0]["message"]["function_call"]["name"],
                    "content": function_content
                })
            
            is_stop = (response["choices"][0]["finish_reason"] == "stop")

        answer = response["choices"][0]["message"]["content"]

        return answer
    
    def call_huggingface_inference_api(self, function_name: str, function_arguments: Union[object, None]) -> Union[str, None]:
        if function_name == "object_detection":
            function_content = self.huggingface_inference_client.post(data=function_arguments["data_path"], model="facebook/detr-resnet-50").text
        elif function_name == "image_to_text":
            function_content = self.huggingface_inference_client.image_to_text(function_arguments["data_path"])
        else:
            function_content = ""

        if self.is_verbose:
            print(f"{function_content = }")
            print()

        return function_content


if __name__ == "__main__":
    function_calling_gpt = FunctionCallingGPT(model="gpt-3.5-turbo-0613", is_verbose=True)

    data_path = os.path.join(os.getenv("PWD"), "image", "savanna.jpg")

    user_content = "Can you count how many objects in this picture? data_path: " + data_path
    answer = function_calling_gpt(user_content)
    print(f"{answer = }")
    print()

    user_content = "Can you describe this picture? data_path: " + data_path
    answer = function_calling_gpt(user_content)
    print(f"{answer = }")
    print()
