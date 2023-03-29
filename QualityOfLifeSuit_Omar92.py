# Developed by Omar - https://github.com/omar92
# https://civitai.com/user/omar92
# discord: Omar92#3374

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

# INSTALLATION CLEANUP (thanks WAS i got this from you)
# Delete legacy nodes
legacy_was_nodes = ['ChatGPT_Omar92.py','LatentUpscaleMultiply_Omar92.py','StringSuit_Omar92.py']
legacy_was_nodes_found = []
f_disp = False
for f in legacy_was_nodes:
    node_path_dir = os.getcwd()+'/ComfyUI/custom_nodes/'
    file = f'{node_path_dir}{f}'
    if os.path.exists(file):
        import zipfile
        if not f_disp:
            print('\033[34mQualityOflife Node Suite:\033[0m Found legacy nodes. Archiving legacy nodes...')
            f_disp = True
        legacy_was_nodes_found.append(file)
if legacy_was_nodes_found:
    from os.path import basename
    archive = zipfile.ZipFile(f'{node_path_dir}QualityOflife_Backup_{round(time.time())}.zip', "w")
    for f in legacy_was_nodes_found:
        archive.write(f, basename(f))
        try:
            os.remove(f)
        except OSError:
            pass
    archive.close()
if f_disp:
    print('\033[34mQualityOflife Node Suite:\033[0m Legacy cleanup complete.')


# region openAITools

class O_ChatGPT:
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
    CATEGORY = "O >>/OpenAI >>"  # Define the category for the node

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


class load_openAI:
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


class openAi_chat_message:
    """
    create chat message for openAI chatGPT
    """
    # Define the input types for the node
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "role": (["user", "assistant", "system"], {"default": "user"}),
                "content": ("STRING", {"multiline": True, "default": "act as prompt generator ,i will give you text and you describe an image that match that text in details, answer with one response only"}),
            }
        }
    # Define the return type of the node
    RETURN_TYPES = ("OPENAI_CHAT_MESSAGES",)
    FUNCTION = "fun"  # Define the function name for the node
    # Define the category for the node
    CATEGORY = "O >>/OpenAI >>/Advanced >>/ChatGPT"

    def fun(self, role, content):
        return (
            {
                "messages": [{"role": role, "content": content, }]
            },
        )


class openAi_chat_message_STR:
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
    CATEGORY = "O >>/OpenAI >>/Advanced >>/ChatGPT"

    def fun(self, role, content):
        return (
            {
                "messages": [{"role": role, "content": content["string"], }]
            },
        )


class openAi_chat_messages_Combine:
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
    CATEGORY = "O >>/OpenAI >>/Advanced >>/ChatGPT"

    def fun(self, message1, message2):
        messages = message1["messages"] + \
            message2["messages"]  # compine messages

        return (
            {
                "messages": messages
            },
        )


class openAi_chat_completion:
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
    CATEGORY = "O >>/OpenAI >>/Advanced >>/ChatGPT"

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


class DebugOpenAIChatMEssages:
    """
    Debug OpenAI Chat Messages
    """
    # Define the input types for the node
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "messages": ("OPENAI_CHAT_MESSAGES", ),
            }
        }
    # Define the return type of the node
    RETURN_TYPES = ()
    FUNCTION = "fun"  # Define the function name for the node
    OUTPUT_NODE = True
    # Define the category for the node
    CATEGORY = "O >>/OpenAI >>/Advanced >>/ChatGPT"

    def fun(self, messages):
        print("DebugOpenAIChatMEssages:", messages["messages"])
        return ()


class DebugOpenAIChatCompletion:
    """
    Debug OpenAI Chat Completion
    """
    # Define the input types for the node
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "completion": ("OPENAI_CHAT_COMPLETION", ),
            }
        }
    # Define the return type of the node
    RETURN_TYPES = ()
    FUNCTION = "fun"  # Define the function name for the node
    OUTPUT_NODE = True
    # Define the category for the node
    CATEGORY = "O >>/OpenAI >>/Advanced >>/ChatGPT"

    def fun(self, completion):
        print("DebugOpenAIChatCompletion:", completion["completion"])
        return ()
# endregion ChatGPT
# region Image


class openAi_Image_create:
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
    CATEGORY = "O >>/OpenAI >>/Advanced >>/Image"

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


class openAi_Image_Edit:
    """
    edit an image using openai 
    """
    # Define the input types for the node
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "openai": ("OPENAI", ),
                "image": ("IMAGE",),
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
    CATEGORY = "O >>/OpenAI >>/Advanced >>/Image"

    def fun(self, openai, image, prompt, number, size):
        # Create a chat completion using the OpenAI module
        openai = openai["openai"]
        prompt = prompt["string"]
        number = 1

    # Convert PyTorch tensor to NumPy array
        image = image[0]
        i = 255. * image.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        # Save the image to a BytesIO object as a PNG file
        with io.BytesIO() as output:
            img.save(output, format='PNG')
            binary_image = output.getvalue()

        # Create a circular mask with alpha 0 in the middle
        mask = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        center = (image.shape[1] // 2, image.shape[0] // 2)
        radius = min(center[0], center[1])
        draw = ImageDraw.Draw(Image.fromarray(mask, mode='RGBA'))
        draw.ellipse((center[0]-radius, center[1]-radius, center[0]+radius,
                     center[1]+radius), fill=(0, 0, 0, 255), outline=(0, 0, 0, 0))
        del draw
        # Save the mask to a BytesIO object as a PNG file
        with io.BytesIO() as output:
            Image.fromarray(mask, mode='RGBA').save(output, format='PNG')
            binary_mask = output.getvalue()

        imagesURLS = openai.Image.create_edit(
            image=binary_image,
            mask=binary_mask,
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
            mask = torch.zeros(
                (1, image.shape[2], image.shape[3]), dtype=torch.float32, device="cpu")
        print("image_tensor: done")
        return (image, mask)


class openAi_Image_variation:
    """
    edit an image using openai 
    """
    # Define the input types for the node
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "openai": ("OPENAI", ),
                "image": ("IMAGE",),
                "number": ("INT", {"default": 1, "min": 1, "max": 10,  "step": 1}),
                "size": (["256x256", "512x512", "1024x1024"], {"default": "256x256"}),
            }
        }
    # Define the return type of the node
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "fun"  # Define the function name for the node
    OUTPUT_NODE = True
    # Define the category for the node
    CATEGORY = "O >>/OpenAI >>/Advanced >>/Image"

    def fun(self, openai, image, number, size):
        # Create a chat completion using the OpenAI module
        openai = openai["openai"]
        number = 1

    # Convert PyTorch tensor to NumPy array
        image = image[0]
        i = 255. * image.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        # Save the image to a BytesIO object as a PNG file
        with io.BytesIO() as output:
            img.save(output, format='PNG')
            binary_image = output.getvalue()

        imagesURLS = openai.Image.create_variation(
            image=binary_image,
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
            mask = torch.zeros(
                (1, image.shape[2], image.shape[3]), dtype=torch.float32, device="cpu")
        print("image_tensor: done")
        return (image, mask)
# endregion Image
# endregion advanced
# endregion openAI

# region latentTools


class LatentUpscaleMultiply:
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
    CATEGORY = "O >>/latent  >>"

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
# endregion

# region StringTools


class O_String:
    """
    this node is a simple string node that can be used to hold userinput as string
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"string": ("STRING", {"multiline": True})}}

    RETURN_TYPES = ("STR",)
    FUNCTION = "ostr"
    CATEGORY = "O >>/string  >>"

    @staticmethod
    def ostr(string):
        return ({"string": string},)


class DebugString:
    """
    This node will write a string to the console
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"string": ("STR",)}}

    RETURN_TYPES = ()
    FUNCTION = "debug_string"
    OUTPUT_NODE = True
    CATEGORY = "O >>/string  >>"

    @staticmethod
    def debug_string(string):
        print("debugString:", string["string"])
        return ()


class string2Image:
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
    CATEGORY = "O >>/string  >>"

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


class CLIPStringEncode:
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

    CATEGORY = "O >>/string  >>"

    def encode(self, string, clip):
        return ([[clip.encode(string["string"]), {}]], )

# region String/operations


class concat_String:
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
    CATEGORY = "O >>/string  >>/operations  >>"

    @staticmethod
    def fun(string1, string2):
        return ({"string": string1["string"] + string2["string"]},)


class trim_String:
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
    CATEGORY = "O >>/string  >>/operations  >>"

    def fun(self, string):
        return (
            {
                "string": (string["string"].strip()),
            },
        )


class replace_String:
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
    CATEGORY = "O >>/string  >>/operations  >>"

    @staticmethod
    def fun(string, old, new):
        return ({"string": string["string"].replace(old, new)},)

 # replace a string with another string


class replace_String_advanced:
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
    CATEGORY = "O >>/string  >>/operations  >>"

    @staticmethod
    def fun(string, old, new):
        return ({"string": string["string"].replace(old["string"], new["string"])},)
# endregion
# endregion


# Define the node class mappings
NODE_CLASS_MAPPINGS = {
    # openAITools
    "ChatGPT _O": O_ChatGPT,
    # openAiTools > Advanced
    "load_openAI _O": load_openAI,
    # openAiTools > Advanced > ChatGPT
    "Chat_Message _O": openAi_chat_message,
    "Chat_Message_fromString _O": openAi_chat_message_STR,
    "compine_chat_messages _O": openAi_chat_messages_Combine,
    "Chat_Completion _O": openAi_chat_completion,
    "debug messages_O": DebugOpenAIChatMEssages,
    "debug Completeion_O": DebugOpenAIChatCompletion,
    # openAiTools > Advanced > image
    "create_image _O": openAi_Image_create,
    #"Edit_image _O": openAi_Image_Edit, # coming soon
    "variation_image _O": openAi_Image_variation,
    # latentTools
    "LatentUpscaleMultiply": LatentUpscaleMultiply,
    # StringTools
    "String _O": O_String,
    "Debug String _O": DebugString,
    "concat Strings _O": concat_String,
    "trim String _O": trim_String,
    "replace String _O": replace_String,
    "replace String advanced _O": replace_String_advanced,
    "string2Image _O": string2Image,
    "CLIPStringEncode _O": CLIPStringEncode,
}
