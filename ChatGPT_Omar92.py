# Developed by Omar - https://github.com/omar92
# https://civitai.com/user/omar92
# discord: Omar92#3374

import importlib

class O_ChatGPT:
    # Define the input types for the node
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),  # Multiline string input for the prompt
                "api_key_file": ("STRING", {"file": True, "default": "api_key.txt"})  # File input for the API key
            }
        }

    RETURN_TYPES = ("STR",)  # Define the return type of the node
    FUNCTION = "fun"  # Define the function name for the node
    CATEGORY = "O/ChatGPT"  # Define the category for the node

    def fun(self, api_key_file, prompt):
        self.install_openai()  # Install the OpenAI module if not already installed
        import openai  # Import the OpenAI module

        api_key = self.get_api_key(api_key_file)  # Get the API key from the file

        openai.api_key = api_key  # Set the API key for the OpenAI module
        
        # Create a chat completion using the OpenAI module
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "act as prompt generator ,i will give you text and you describe an image that match that text in details, answer with one response only"},
                {"role": "user", "content": prompt}
            ]
        )       
        answer = completion["choices"][0]["message"]["content"]  # Get the answer from the chat completion

        return (
            {
                "string": answer,  # Return the answer as a string
            },
        )

    # Helper function to get the API key from the file
    def get_api_key(self, api_key_file):
        custom_nodes_dir = 'ComfyUI/custom_nodes/'  # Define the directory for the file
        with open(custom_nodes_dir+api_key_file, 'r') as f:  # Open the file and read the API key
            api_key = f.read().strip()
        return api_key  # Return the API key

    # Helper function to install the OpenAI module if not already installed
    def install_openai(self):
        try:
            importlib.import_module('openai')
        except ImportError:
            import pip
            pip.main(['install', 'openai'])

# Define the node class mappings
NODE_CLASS_MAPPINGS = {
    "ChatGPT _O": O_ChatGPT,
}
