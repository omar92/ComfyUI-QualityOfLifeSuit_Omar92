# Developed by Omar - https://github.com/omar92
# https://civitai.com/user/omar92
# discord: Omar92#3374

###
#
#  All nodes in this file are deprecated and will be removed in the future , i left them here for backward compatibility
#
###

import io
import os
import time
import numpy as np
import requests
import torch
from PIL import Image, ImageFont, ImageDraw
from PIL import Image, ImageDraw
import importlib
import comfy.samplers
import comfy.sd
import comfy.utils

# -------------------------------------------------------------------------------------------
# region deprecated
# region openAITools


class O_ChatGPT_deprecated:
    """
    this node is based on the openAI GPT-3 API to generate propmpts using the AI
    """
    # Define the input types for the node
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Multiline string input for the prompt
                "prompt": ("STRING", {"multiline": True}),
                # File input for the API key
                "api_key_file": ("STRING", {"file": True, "default": "api_key.txt"})
            }
        }

    RETURN_TYPES = ("STR",)  # Define the return type of the node
    FUNCTION = "fun"  # Define the function name for the node
    CATEGORY = "O >>/deprecated >>/OpenAI >>"  # Define the category for the node

    def fun(self, api_key_file, prompt):
        self.install_openai()  # Install the OpenAI module if not already installed
        import openai  # Import the OpenAI module

        # Get the API key from the file
        api_key = self.get_api_key(api_key_file)

        openai.api_key = api_key  # Set the API key for the OpenAI module

        # Create a chat completion using the OpenAI module
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "act as prompt generator ,i will give you text and you describe an image that match that text in details, answer with one response only"},
                {"role": "user", "content": prompt}
            ]
        )
        # Get the answer from the chat completion
        answer = completion["choices"][0]["message"]["content"]

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
# region advanced

    """
    this node will load  openAI model
    """
    # Define the input types for the node
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # File input for the API key
                "api_key_file": ("STRING", {"file": True, "default": "api_key.txt"})
            }
        }
    RETURN_TYPES = ("OPENAI",)  # Define the return type of the node
    FUNCTION = "fun"  # Define the function name for the node
    CATEGORY = "O >>/OpenAI >>/Advanced >>"  # Define the category for the node

    def fun(self, api_key_file):
        self.install_openai()  # Install the OpenAI module if not already installed
        import openai  # Import the OpenAI module

        # Get the API key from the file
        api_key = self.get_api_key(api_key_file)
        openai.api_key = api_key  # Set the API key for the OpenAI module

        return (
            {
                "openai": openai,  # Return openAI model
            },
        )

    # Helper function to install the OpenAI module if not already installed
    def install_openai(self):
        try:
            importlib.import_module('openai')
        except ImportError:
            import pip
            pip.main(['install', 'openai'])

    # Helper function to get the API key from the file
    def get_api_key(self, api_key_file):
        custom_nodes_dir = 'ComfyUI/custom_nodes/'  # Define the directory for the file
        with open(custom_nodes_dir+api_key_file, 'r') as f:  # Open the file and read the API key
            api_key = f.read().strip()
        return api_key  # Return the API key
# region ChatGPT


class openAi_chat_message_STR_deprecated:
    """
    create chat message for openAI chatGPT
    """
    # Define the input types for the node
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "role": (["user", "assistant", "system"], {"default": "user"}),
                "content": ("STR",),
            }
        }
    # Define the return type of the node
    RETURN_TYPES = ("OPENAI_CHAT_MESSAGES",)
    FUNCTION = "fun"  # Define the function name for the node
    # Define the category for the node
    CATEGORY = "O >>/deprecated >>/OpenAI >>/Advanced >>/ChatGPT"

    def fun(self, role, content):
        return (
            {
                "messages": [{"role": role, "content": content["string"], }]
            },
        )
# endregion ChatGPT
# region Image

class openAi_chat_messages_Combine_deprecated:
    """
     compine chat messages into 1 tuple
    """
    # Define the input types for the node
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "message1": ("OPENAI_CHAT_MESSAGES", ),
                "message2": ("OPENAI_CHAT_MESSAGES", ),
            }
        }
    # Define the return type of the node
    RETURN_TYPES = ("OPENAI_CHAT_MESSAGES",)
    FUNCTION = "fun"  # Define the function name for the node
    # Define the category for the node
    CATEGORY = "O >>/deprecated >>/OpenAI >>/Advanced >>/ChatGPT >>"

    def fun(self, message1, message2):
        messages = message1["messages"] + \
            message2["messages"]  # compine messages

        return (
            {
                "messages": messages
            },
        )
class openAi_Image_create_deprecated:
    """
    create image using openai 
    """
    # Define the input types for the node
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "openai": ("OPENAI", ),
                "prompt": ("STR",),
                "number": ("INT", {"default": 1, "min": 1, "max": 10,  "step": 1}),
                "size": (["256x256", "512x512", "1024x1024"], {"default": "256x256"}),
            }
        }
    # Define the return type of the node
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "fun"  # Define the function name for the node
    OUTPUT_NODE = True
    # Define the category for the node
    CATEGORY = "O >>/deprecated >>/OpenAI >>/Advanced >>/Image"

    def fun(self, openai, prompt, number, size):
        # Create a chat completion using the OpenAI module
        openai = openai["openai"]
        prompt = prompt["string"]
        number = 1
        imagesURLS = openai.Image.create(
            prompt=prompt,
            n=number,
            size=size
        )
        imageURL = imagesURLS["data"][0]["url"]
        print("imageURL:", imageURL)
        image = requests.get(imageURL).content
        i = Image.open(io.BytesIO(image))
        image = i.convert("RGBA")
        image = np.array(image).astype(np.float32) / 255.0
        # image_np = np.transpose(image_np, (2, 0, 1))
        image = torch.from_numpy(image)[None,]
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
        print("image_tensor: done")
        return (image, mask)


class openAi_chat_completion_deprecated:
    """
    create chat completion for openAI chatGPT
    """
    # Define the input types for the node
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "openai": ("OPENAI", ),
                "model": ("STRING", {"multiline": False, "default": "gpt-3.5-turbo"}),
                "messages": ("OPENAI_CHAT_MESSAGES", ),
            }
        }
    # Define the return type of the node
    RETURN_TYPES = ("STR", "OPENAI_CHAT_COMPLETION",)
    FUNCTION = "fun"  # Define the function name for the node
    OUTPUT_NODE = True
    # Define the category for the node
    CATEGORY = "O >>/deprecated >>/OpenAI >>/Advanced >>/ChatGPT"

    def fun(self, openai, model, messages):
        # Create a chat completion using the OpenAI module
        openai = openai["openai"]
        completion = openai.ChatCompletion.create(
            model=model,
            messages=messages["messages"]
        )
        # Get the answer from the chat completion
        content = completion["choices"][0]["message"]["content"]
        return (
            {
                "string": content,  # Return the answer as a string
            },
            {
                "completion": completion,  # Return the chat completion
            }
        )


# endregion Image
# endregion advanced
# endregion openAI
# region StringTools


class O_String_deprecated:
    """
    this node is a simple string node that can be used to hold userinput as string
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"string": ("STRING", {"multiline": True})}}

    RETURN_TYPES = ("STR",)
    FUNCTION = "ostr"
    CATEGORY = "O >>/deprecated >>/string >>"

    @staticmethod
    def ostr(string):
        return ({"string": string},)


class DebugString_deprecated:
    """
    This node will write a string to the console
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"string": ("STR",)}}

    RETURN_TYPES = ()
    FUNCTION = "debug_string"
    OUTPUT_NODE = True
    CATEGORY = "O >>/deprecated >>/string >>"

    @staticmethod
    def debug_string(string):
        print("debugString:", string["string"])
        return ()


class string2Image_deprecated:
    """
    This node will convert a string to an image
    """

    def __init__(self):
        self.font_filepath = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "fonts")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "string": ("STR",),
                "font": ("STRING", {"default": "CALIBRI.TTF", "multiline": False}),
                "size": ("INT", {"default": 36, "min": 0, "max": 255, "step": 1}),
                "font_R": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                "font_G": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                "font_B": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                "background_R": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
                "background_G": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
                "background_B": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "create_image"
    OUTPUT_NODE = False
    CATEGORY = "O >>/deprecated >>/string >>"

    def create_image(self, string, font, size, font_R, font_G, font_B, background_R, background_G, background_B):
        font_color = (font_R, font_G, font_B)
        font = ImageFont.truetype(self.font_filepath+"\\"+font, size)
        mask_image = font.getmask(string["string"], "L")
        image = Image.new("RGBA", mask_image.size,
                          (background_R, background_G, background_B))
        # need to use the inner `img.im.paste` due to `getmask` returning a core
        image.im.paste(font_color, (0, 0) + mask_image.size, mask_image)

        # Convert the  PIL Image to a tensor
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).unsqueeze(0)
        return (image_tensor,)


class CLIPStringEncode_deprecated:
    """
    This node will encode a string with CLIP
    """
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "string": ("STR",),
            "clip": ("CLIP", )
        }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"

    CATEGORY = "O >>/deprecated >>/string >>"

    def encode(self, string, clip):
        return ([[clip.encode(string["string"]), {}]], )
# region String/operations


class concat_String_deprecated:
    """
    This node will concatenate two strings together
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "string1": ("STR",),
            "string2": ("STR",)
        }}

    RETURN_TYPES = ("STR",)
    FUNCTION = "fun"
    CATEGORY = "O >>/deprecated >>/string >>/operations >>"

    @staticmethod
    def fun(string1, string2):
        return ({"string": string1["string"] + string2["string"]},)


class trim_String_deprecated:
    """
    This node will trim a string from the left and right
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "string": ("STR",),
        }}

    RETURN_TYPES = ("STR",)
    FUNCTION = "fun"
    CATEGORY = "O >>/deprecated >>/string >>/operations >>"

    def fun(self, string):
        return (
            {
                "string": (string["string"].strip()),
            },
        )


class replace_String_deprecated:
    """
    This node will replace a string with another string
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "string": ("STR",),
            "old": ("STRING", {"multiline": False}),
            "new": ("STRING", {"multiline": False})
        }}

    RETURN_TYPES = ("STR",)
    FUNCTION = "fun"
    CATEGORY = "O >>/deprecated >>/string >>/operations >>"

    @staticmethod
    def fun(string, old, new):
        return ({"string": string["string"].replace(old, new)},)

 # replace a string with another string


class replace_String_advanced_deprecated:
    """
    This node will replace a string with another string
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "string": ("STR",),
            "old": ("STR",),
            "new": ("STR",),
        }}

    RETURN_TYPES = ("STR",)
    FUNCTION = "fun"
    CATEGORY = "O >>/deprecated >>/string >>/operations >>"

    @staticmethod
    def fun(string, old, new):
        return ({"string": string["string"].replace(old["string"], new["string"])},)
# endregion
# endregion


class LatentUpscaleMultiply_deprecated:
    """
    Upscale the latent code by multiplying the width and height by a factor
    """
    upscale_methods = ["nearest-exact", "bilinear", "area"]
    crop_methods = ["disabled", "center"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "upscale_method": (cls.upscale_methods,),
                "WidthMul": ("FLOAT", {"default": 1.25, "min": 0.0, "max": 10.0, "step": 0.1}),
                "HeightMul": ("FLOAT", {"default": 1.25, "min": 0.0, "max": 10.0, "step": 0.1}),
                "crop": (cls.crop_methods,),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "upscale"
    CATEGORY = "O >>/deprecated >>/latent >>"

    def upscale(self, samples, upscale_method, WidthMul, HeightMul, crop):
        s = samples.copy()
        x = samples["samples"].shape[3]
        y = samples["samples"].shape[2]

        new_x = int(x * WidthMul)
        new_y = int(y * HeightMul)
        print(f"upscale from ({x*8},{y*8}) to ({new_x*8},{new_y*8})")

        def enforce_mul_of_64(d):
            leftover = d % 8
            if leftover != 0:
                d += 8 - leftover
            return d

        s["samples"] = comfy.utils.common_upscale(
            samples["samples"], enforce_mul_of_64(
                new_x), enforce_mul_of_64(new_y), upscale_method, crop
        )
        return (s,)

# endregion deprecated


# Define the node class mappings
NODE_CLASS_MAPPINGS = {
    # deprecated
    "ChatGPT _O": O_ChatGPT_deprecated,
    "Chat_Message_fromString _O": openAi_chat_message_STR_deprecated,
    "compine_chat_messages _O": openAi_chat_messages_Combine_deprecated,
    "Chat_Completion _O": openAi_chat_completion_deprecated,
    "create_image _O": openAi_Image_create_deprecated,
    "String _O": O_String_deprecated,
    "Debug String _O": DebugString_deprecated,
    "concat Strings _O": concat_String_deprecated,
    "trim String _O": trim_String_deprecated,
    "replace String _O": replace_String_deprecated,
    "replace String advanced _O": replace_String_advanced_deprecated,
    "string2Image _O": string2Image_deprecated,
    "CLIPStringEncode _O": CLIPStringEncode_deprecated,
    "CLIPStringEncode _O": CLIPStringEncode_deprecated,
    "LatentUpscaleMultiply": LatentUpscaleMultiply_deprecated,
}
