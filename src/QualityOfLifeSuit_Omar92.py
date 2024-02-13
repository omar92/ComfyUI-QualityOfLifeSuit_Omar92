# Developed by Omar - https://github.com/omar92
# https://civitai.com/user/omar92
# discord: Omar92#3374

import io
import json
import os
import random
import time
from urllib.request import urlopen
import numpy as np
import requests
import torch
from PIL import Image, ImageFont, ImageDraw
import importlib
import comfy.samplers
import comfy.sd
import comfy.utils
import torch.nn as nn

MAX_RESOLUTION = 8192

# region INSTALLATION CLEANUP (thanks WAS i got this from you)
# Delete legacy nodes
legacy_nodes = ['ChatGPT_Omar92.py',
                'LatentUpscaleMultiply_Omar92.py', 'StringSuit_Omar92.py']
legacy_nodes_found = []
f_disp = False
for f in legacy_nodes:
    node_path_dir = os.getcwd()+'./custom_nodes/'
    file = f'{node_path_dir}{f}'
    if os.path.exists(file):
        import zipfile
        if not f_disp:
            print(
                '\033[33mQualityOflife Node Suite:\033[0m Found legacy nodes. Archiving legacy nodes...')
            f_disp = True
        legacy_nodes_found.append(file)
if legacy_nodes_found:
    from os.path import basename
    archive = zipfile.ZipFile(
        f'{node_path_dir}QualityOflife_Backup_{round(time.time())}.zip', "w")
    for f in legacy_nodes_found:
        archive.write(f, basename(f))
        try:
            os.remove(f)
        except OSError:
            pass
    archive.close()
if f_disp:
    print('\033[33mQualityOflife Node Suite:\033[0m Legacy cleanup complete.')
# endregion

# region global
PACKAGE_NAME = '\033[33mQualityOfLifeSuit_Omar92:\033[0m'
NODE_FILE = os.path.abspath(__file__)
SUIT_DIR = (os.path.dirname(os.path.dirname(NODE_FILE))
            if os.path.dirname(os.path.dirname(NODE_FILE)) == 'QualityOfLifeSuit_Omar92'
            or os.path.dirname(os.path.dirname(NODE_FILE)) == 'QualityOfLifeSuit_Omar92-dev'
            else os.path.dirname(NODE_FILE))
SUIT_DIR = os.path.normpath(os.path.join(SUIT_DIR, '..'))
print(f'\033[33mQualityOfLifeSuit_Omar92_DIR:\033[0m {SUIT_DIR}')


def enforce_mul_of_64(d):
    leftover = d % 8          # 8 is the number of pixels per byte
    if leftover != 0:         # if the number of pixels is not a multiple of 8
        if (leftover < 4):       # if the number of pixels is less than 4
            d -= leftover     # remove the leftover pixels
        else:                 # if the number of pixels is more than 4
            d += 8 - leftover  # add the leftover pixels

    return d

# endregion


# region openAITools


def install_openai():
    # Helper function to install the OpenAI module if not already installed
    try:
        importlib.import_module('openai')
    except ImportError:
        import pip
        pip.main(['install', 'openai'])


def get_api_key():
    # Helper function to get the API key from the file
    try:
        # open config file
        configPath = os.path.join(SUIT_DIR, "config.json")
        with open(configPath, 'r') as f:  # Open the file and read the API key
            config = json.load(f)
        api_key = config["openAI_API_Key"]
    except:
        print("Error: OpenAI API key file not found OpenAI features wont work for you")
        return ""
    return api_key  # Return the API key


openAI_models = None
#region chatGPTDefaultInitMessages
chatGPTDefaultInitMessage_tags = """
First, some basic Stable Diffusion prompting rules for you to better understand the syntax. The parentheses are there for grouping prompt words together, so that we can set uniform weight to multiple words at the same time. Notice the ":1.2" in (masterpiece, best quality, absurdres:1.2), it means that we set the weight of both "masterpiece" and "best quality" to 1.2. The parentheses can also be used to directly increase weight for single word without adding ":WEIGHT". For example, we can type ((masterpiece)), this will increase the weight of "masterpiece" to 1.21. This basic rule is imperative that any parentheses in a set of prompts have purpose, and so they must not be remove at any case. Conversely, when brackets are used in prompts, it means to decrease the weight of a word. For example, by typing "[bird]", we decrease the weight of the word "bird" by 1.1.
Now, I've develop a prompt template to use generate character portraits in Stable Diffusion. Here's how it works. Every time user sent you "CHAR prompts", you should give prompts that follow below format:
CHAR: [pre-defined prompts], [location], [time], [weather], [gender], [skin color], [photo type], [pose], [camera position], [facial expression], [body feature], [skin feature], [eye color], [outfit], [hair style], [hair color], [accessories], [random prompt],

[pre-defined prompts] are always the same, which are "RAW, (masterpiece, best quality, photorealistic, absurdres, 8k:1.2), best lighting, complex pupils, complex textile, detailed background". Don't change anything in [pre-defined prompts], meaning that you SHOULD NOT REMOVE OR MODIFY the parentheses since their purpose is for grouping prompt words together so that we can set uniform weight to them;
[location] is the location where character is in, can be either outdoor location or indoor, but need to be specific;
[time] refers to the time of day, can be "day", "noon", "night", "evening", "dawn" or "dusk";
[weather] is the weather, for example "windy", "rainy" or "cloudy";
[gender] is either "1boy" or "1girl";
[skin color] is the skin color of the character, could be "dark skin", "yellow skin" or "pale skin";
[photo type] can be "upper body", "full body", "close up", "mid-range", "Headshot", "3/4 shot" or "environmental portrait";
[pose] is the character's pose, for example, "standing", "sitting", "kneeling" or "squatting" ...;
[camera position] can be "from top", "from below", "from side", "from front" or "from behind";
[facial expression] is the expression of the character, you should give user a random expression;
[body feature] describe how the character's body looks like, for example, it could be "wide hip", "large breasts" or "sexy", try to be creative;
[skin feature] is the feature of character's skin. Could be "scar on skin", "dirty skin", "tanned mark", "birthmarks" or other skin features you can think of;
[eye color] is the pupil color of the character, it can be of any color as long as the color looks natural on human eyes, so avoid colors like pure red or pure black;
[outfit] is what character wears, it should include at least the top wear, bottom wear and footwear, for example, "crop top, shorts, sneakers", the style of outfit can be any, but the [character gender] should be considered;
[hair style] is the hairstyle of the character, [character gender] should be taken into account when setting the hairstyle;
[hair color] can be of any color, for example, "orange hair", "multi-colored hair";
[accessories] is the accessory the character might wear, can be "chocker", "earrings", "bracelet" or other types of accessory;
[random prompt] will test your creativity, put anything here, just remember that you can only use nouns in [random prompt], the number of [random prompt] can be between 1 to 4. For example, you could give "campfire", but you can also give "shooting star, large moon, fallen leaves". Again, be creative with this one.

also use gelbooru  tags as much as you can
if you use gelbooru  write "gTags" before it 
Do not use markdown syntax in prompts, do not use capital letter and keep all prompt words in the same line. Respond with "prompt:" to start prompting with us.

""";

chatGPTDefaultInitMessage_description = """
act as prompt generator ,i will give you text and you describe an image that match that text in details use gelbooru  tags in your description also describe the high quality of the image, answer with one response only 
""";
def get_init_message(isTags=False):
    if(isTags):
        return chatGPTDefaultInitMessage_tags
    else:
        return chatGPTDefaultInitMessage_description

#endregion chatGPTDefaultInitMessages
def get_openAI_models():
    global openAI_models
    if (openAI_models != None):
        return openAI_models

    install_openai()
    from openai import OpenAI
    # Get the API key from the file
    api_key = get_api_key()
    client = OpenAI(
                # This is the default and can be omitted
                api_key=api_key,
            )

    try:
        models = client.models.list()  # Get the list of models
        print("all models:",models)
    except:
        print("Error: OpenAI API key is invalid OpenAI features wont work for you")
        return []

    openAI_models = []  # Create a list for the chat models
    for model in getattr(models, "data"):  # Loop through the models
        openAI_models.append(getattr(model, "id"))  # Add the model to the list

    return openAI_models  # Return the list of chat models


openAI_gpt_models = None


def get_gpt_models():
    global openAI_gpt_models
    if (openAI_gpt_models != None):
        return openAI_gpt_models
    models = get_openAI_models()
    openAI_gpt_models = ["gpt-3.5-turbo"]  # Create a list for the chat models
    for model in models:  # Loop through the models
        if ("gpt" in model.lower()):
            openAI_gpt_models.append(model)

    return openAI_gpt_models  # Return the list of chat models


class O_ChatGPT_O:
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
                "model": (get_gpt_models(), {"default": "gpt-3.5-turbo"}),
                "behaviour": (["tags","description"], {"default": "description"}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("STRING",)  # Define the return type of the node
    FUNCTION = "fun"  # Define the function name for the node
    CATEGORY = "O/OpenAI"  # Define the category for the node

    def fun(self,  model, prompt,behaviour, seed):
        install_openai()  # Install the OpenAI module if not already installed
        #import openai  # Import the OpenAI module
        from openai import OpenAI
        # Get the API key from the file
        api_key = get_api_key()
        client = OpenAI(
                    # This is the default and can be omitted
                    api_key=api_key,
                )
        #openai.api_key = api_key  # Set the API key for the OpenAI module
        initMessage = "";
        if(behaviour == "description"):
            initMessage = get_init_message(False);
        else:
            initMessage = get_init_message(True);
        # Create a chat completion using the OpenAI module
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content":initMessage},
                    {"role": "user", "content": prompt}
                ]
            )
        except:  # sometimes it fails first time to connect to server
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": initMessage},
                    {"role": "user", "content": prompt}
                ]
            )
        # Get the answer from the chat completion
        answer = completion.choices[0].message.content
        return (answer,)  # Return the answer as a string


class O_ChatGPT_medium_O:
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
                "initMsg": ("STRING", {"multiline": True, "default": get_init_message()}),
                "model": (get_gpt_models(), {"default": "gpt-3.5-turbo"}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("STRING",)  # Define the return type of the node
    FUNCTION = "fun"  # Define the function name for the node
    CATEGORY = "O/OpenAI"  # Define the category for the node

    def fun(self,  model, prompt, initMsg, seed):
        install_openai()  # Install the OpenAI module if not already installed
        #import openai  # Import the OpenAI module
        from openai import OpenAI
        # Get the API key from the file
        api_key = get_api_key()

        #openai.api_key = api_key  # Set the API key for the OpenAI module
        client = OpenAI(
            # This is the default and can be omitted
            api_key=api_key,
        )
        # Create a chat completion using the OpenAI module
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": initMsg},
                    {"role": "user", "content": prompt}
                ]
            )
        except:  # sometimes it fails first time to connect to server
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": initMsg},
                    {"role": "user", "content": prompt}
                ]
            )
        # Get the answer from the chat completion
        answer = completion.choices[0].message.content
        return (answer,)  # Return the answer as a string


# region advanced


class load_openAI_O:
    """
    this node will load  openAI model
    """
    # Define the input types for the node
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
            }
        }
    RETURN_TYPES = ("CLIENT",)  # Define the return type of the node
    FUNCTION = "fun"  # Define the function name for the node
    CATEGORY = "O/OpenAI/Advanced"  # Define the category for the node

    def fun(self):
        install_openai()  # Install the OpenAI module if not already installed
        from openai import OpenAI
        # Get the API key from the file
        api_key = get_api_key()
        client = OpenAI(
                    # This is the default and can be omitted
                    api_key=api_key,
                )
        
        return (
            {
                "client": client,  # Return openAI model
            },
        )
# region ChatGPT


class openAi_chat_message_O:
    """
    create chat message for openAI chatGPT
    """
    # Define the input types for the node
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "role": (["user", "assistant", "system"], {"default": "user"}),
                "content": ("STRING", {"multiline": True, "default":get_init_message()}),
            }
        }
    # Define the return type of the node
    RETURN_TYPES = ("OPENAI_CHAT_MESSAGES",)
    FUNCTION = "fun"  # Define the function name for the node
    # Define the category for the node
    CATEGORY = "O/OpenAI/Advanced/ChatGPT"

    def fun(self, role, content):
        return (
            {
                "messages": [{"role": role, "content": content, }]
            },
        )


class openAi_chat_messages_Combine_O:
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
    CATEGORY = "O/OpenAI/Advanced/ChatGPT"

    def fun(self, message1, message2):
        messages = message1["messages"] + \
            message2["messages"]  # compine messages

        return (
            {
                "messages": messages
            },
        )


class openAi_chat_completion_O:
    """
    create chat completion for openAI chatGPT
    """
    # Define the input types for the node
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "client": ("CLIENT", ),
                # "model": ("STRING", {"multiline": False, "default": "gpt-3.5-turbo"}),
                "model": (get_gpt_models(), {"default": "gpt-3.5-turbo"}),
                "messages": ("OPENAI_CHAT_MESSAGES", ),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    # Define the return type of the node
    RETURN_TYPES = ("STRING", "OPENAI_CHAT_COMPLETION",)
    FUNCTION = "fun"  # Define the function name for the node
    OUTPUT_NODE = True
    # Define the category for the node
    CATEGORY = "O/OpenAI/Advanced/ChatGPT"

    def fun(self, client, model, messages, seed):
        # Create a chat completion using the OpenAI module
        client = client["client"]
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages["messages"]
            )
        except:  # sometimes it fails first time to connect to server
            completion = client.chat.completions.create(
                model=model,
                messages=messages["messages"]
            )
        # Get the answer from the chat completion
        content = completion.choices[0].message.content
        return (
            content,  # Return the answer as a string
            completion,  # Return the chat completion
        )


class DebugOpenAIChatMEssages_O:
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
    CATEGORY = "O/debug/OpenAI/Advanced/ChatGPT"

    def fun(self, messages):
        print(f'{PACKAGE_NAME}:OpenAIChatMEssages', messages["messages"])
        return ()


class DebugOpenAIChatCompletion_O:
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
    CATEGORY = "O/debug/OpenAI/Advanced/ChatGPT"

    def fun(self, completion):
        print(f'{PACKAGE_NAME}:OpenAIChatCompletion:', completion)
        return ()
# endregion ChatGPT
# region Image


class openAi_Image_create_O:
    """
    create image using openai
    """
    # Define the input types for the node
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "client": ("CLIENT", ),
                "prompt": ("STRING", {"multiline": True}),
                "number": ("INT", {"default": 1, "min": 1, "max": 10,  "step": 1}),
                "size": (["256x256", "512x512", "1024x1024"], {"default": "256x256"}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    # Define the return type of the node
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "fun"  # Define the function name for the node
    OUTPUT_NODE = True
    # Define the category for the node
    CATEGORY = "O/OpenAI/Advanced/Image"

    def fun(self, client, prompt, number, size, seed):
        # Create a chat completion using the OpenAI module
        client = client["client"]
        prompt = prompt
        number = 1

        imageURL = ""
        try:
            imagesURLS = client.images.generate(
                prompt=prompt,
                n=number,
                size=size
            )
            imageURL = imagesURLS["data"][0]["url"]
        except Exception as e:
            print(f'{PACKAGE_NAME}:openAi_Image_create_O:', e)
            imageURL = "https://i.imgur.com/removed.png"

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
        return (image, mask)


class openAi_Image_Edit_O:
    """
    edit an image using openai
    """
    # Define the input types for the node
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "client": ("CLIENT", ),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True}),
                "number": ("INT", {"default": 1, "min": 1, "max": 10,  "step": 1}),
                "size": (["256x256", "512x512", "1024x1024"], {"default": "256x256"}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    # Define the return type of the node
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "fun"  # Define the function name for the node
    OUTPUT_NODE = True
    # Define the category for the node
    CATEGORY = "O/OpenAI/Advanced/Image"

    def fun(self, client, image, prompt, number, size, seed):
        # Create a chat completion using the OpenAI module
        client = client["client"]
        prompt = prompt
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

        imageURL = ""
        try:
            imagesURLS = client.images.edit(
                image=binary_image,
                mask=binary_mask,
                prompt=prompt,
                n=number,
                size=size
            )
            imageURL = imagesURLS["data"][0]["url"]
        except Exception as e:
            print(f'{PACKAGE_NAME}:openAi_Image_create_O:', e)
            imageURL = "https://i.imgur.com/removed.png"

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
        return (image, mask)


class openAi_Image_variation_O:
    """
    edit an image using openai
    """
    # Define the input types for the node
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "client": ("CLIENT", ),
                "image": ("IMAGE",),
                "number": ("INT", {"default": 1, "min": 1, "max": 10,  "step": 1}),
                "size": (["256x256", "512x512", "1024x1024"], {"default": "256x256"}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    # Define the return type of the node
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "fun"  # Define the function name for the node
    OUTPUT_NODE = True
    # Define the category for the node
    CATEGORY = "O/OpenAI/Advanced/Image"

    def fun(self, client, image, number, size, seed):
        # Create a chat completion using the OpenAI module
        client = client["client"]
        number = 1

    # Convert PyTorch tensor to NumPy array
        image = image[0]
        i = 255. * image.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        # Save the image to a BytesIO object as a PNG file
        with io.BytesIO() as output:
            img.save(output, format='PNG')
            binary_image = output.getvalue()

        imageURL = " "
        try:
            imagesURLS = client.images.create_variation(
                image=binary_image,
                n=number,
                size=size
            )
            imageURL = imagesURLS["data"][0]["url"]
        except Exception as e:
            print(f'{PACKAGE_NAME}:openAi_Image_create_O:', e)
            imageURL = "https://i.imgur.com/removed.png"

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
        return (image, mask)
# endregion Image
# endregion advanced
# endregion openAI


# region latentTools


class LatentUpscaleFactor_O:
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
                "WidthFactor": ("FLOAT", {"default": 1.25, "min": 0.0, "max": 10.0, "step": 0.28125}),
                "HeightFactor": ("FLOAT", {"default": 1.25, "min": 0.0, "max": 10.0, "step":  0.28125}),
                "crop": (cls.crop_methods,),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "upscale"
    CATEGORY = "O/latent"

    def upscale(self, samples, upscale_method, WidthFactor, HeightFactor, crop):
        s = samples.copy()
        x = samples["samples"].shape[3]
        y = samples["samples"].shape[2]

        new_x = int(x * WidthFactor)
        new_y = int(y * HeightFactor)

        if (new_x > MAX_RESOLUTION):
            new_x = MAX_RESOLUTION
        if (new_y > MAX_RESOLUTION):
            new_y = MAX_RESOLUTION

        print(f'{PACKAGE_NAME}:upscale from ({x*8},{y*8}) to ({new_x*8},{new_y*8})')

        s["samples"] = comfy.utils.common_upscale(
            samples["samples"], enforce_mul_of_64(
                new_x), enforce_mul_of_64(new_y), upscale_method, crop
        )
        return (s,)


class LatentUpscaleFactorSimple_O:
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
                "factor": ("FLOAT", {"default": 1.25, "min": 0.0, "max": 10.0, "step": 0.28125}),
                "crop": (cls.crop_methods,),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "upscale"
    CATEGORY = "O/latent"

    def upscale(self, samples, upscale_method, factor, crop):
        s = samples.copy()
        x = samples["samples"].shape[3]
        y = samples["samples"].shape[2]

        new_x = int(x * factor)
        new_y = int(y * factor)

        if (new_x > MAX_RESOLUTION):
            new_x = MAX_RESOLUTION
        if (new_y > MAX_RESOLUTION):
            new_y = MAX_RESOLUTION

        print(f'{PACKAGE_NAME}:upscale from ({x*8},{y*8}) to ({new_x*8},{new_y*8})')

        s["samples"] = comfy.utils.common_upscale(
            samples["samples"], enforce_mul_of_64(
                new_x), enforce_mul_of_64(new_y), upscale_method, crop
        )
        return (s,)


class SelectLatentImage_O:
    """
    Select a single image from a batch of generated latent images.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "index": ("INT", {"default": 0, "min": 0}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "fun"
    CATEGORY = "O/latent"

    def fun(self, samples, index):
        # Get the batch size and number of channels
        batch_size, num_channels, height, width = samples["samples"].shape

        # Ensure that the index is within bounds
        if index >= batch_size:
            index = batch_size - 1

        # Select the specified image
        selected_image = samples["samples"][index].unsqueeze(0)

        # Return the selected image
        return ({"samples": selected_image},)


class VAEDecodeParallel_O:
    def __init__(self, device="cpu"):
        self.device = device
        self.device_count = torch.cuda.device_count() if device != "cpu" else 1
        self.module = VAEDecodeOriginal(device)
        self.net = nn.DataParallel(self.module)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"samples": ("LATENT", ), "vae": ("VAE", )}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode_parallel"
    CATEGORY = "latent"

    def decode_parallel(self, vae, samples):
        batch_size = samples["samples"].shape[0]
        images = torch.zeros((batch_size, 3, 256, 256)).to(self.device)

        for i in range(0, batch_size, self.device_count):
            batch_samples = samples["samples"][i:i +
                                               self.device_count].to(self.device)
            batch_images = self.net(vae, {"samples": batch_samples})[
                0].to(self.device)
            images[i:i+self.device_count] = batch_images

        return (images,)


class VAEDecodeOriginal:
    def __init__(self, device="cpu"):
        self.device = device

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"samples": ("LATENT", ), "vae": ("VAE", )}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"

    CATEGORY = "latent"

    def decode(self, vae, samples):
        return (vae.decode(samples["samples"]), )

# endregion latentTools

# region TextTools


class seed2String_O:
    """
    This node convert seeds to string // can be used to force the system to read a string again if it got compined with it 
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"seed": ("SEED")}}

    RETURN_TYPES = ("STRING")
    FUNCTION = "fun"
    CATEGORY = "O/utils"

    def fun(self, seed):
        return (str(seed))


class saveTextToFile_O:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": '', "multiline": False, "defaultBehavior": "input"}),
                "filename": ("STRING", {"default": "log.txt", "multiline": False}),
            },
            "optional": {
                "append": (["true", "false"], {"default": True})
            }
        }

    OUTPUT_NODE = True
    RETURN_TYPES = ()
    FUNCTION = "fun"
    CATEGORY = "O/text"

    def fun(self, text, filename, append):
        # append dateTime
        current_time = time.strftime("%d/%m/%Y %H:%M:%S")  # dd/mm/YY H:M:S
        textToSave = f'{current_time}:  \n'
        # append text in new line
        textToSave += f' {text}  \n\n'

        self.saveTextToFile(textToSave, filename, append)

        return (textToSave, )

    def saveTextToFile(self, text, filename, append):
        saveDir = os.path.join(SUIT_DIR, "output")
        saveFile = os.path.join(saveDir, filename)

        # Create directory if it does not exist
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)

        # Write to file
        mode = "a" if append else "w"
        try:
            with open(saveFile, mode, encoding="utf-8") as f:
                f.write(text)
        except OSError as e:
            print(f'{PACKAGE_NAME}:error writing to file {saveFile}')


fonts = None


def loadFonts():

    global fonts
    if (fonts != None):
        return fonts
    try:
        fonts_filepath = os.path.join(SUIT_DIR, "fonts")
        fonts = []
        for file in os.listdir(fonts_filepath):
            if file.endswith(".ttf") or file.endswith(".otf") or file.endswith(".ttc") or file.endswith(".TTF") or file.endswith(".OTF") or file.endswith(".TTC"):
                fonts.append(file)
    except:
        fonts = []

        if (len(fonts) == 0):
            print(f'{PACKAGE_NAME}:no fonts found in {fonts_filepath}')
            fonts = ["Arial.ttf"]
    return fonts


class Text2Image_O:
    """
    This node will convert a string to an image
    """

    def __init__(self):
        self.font_filepath = os.path.join(SUIT_DIR, "fonts")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "font": (loadFonts(), {"default": loadFonts()[0], }),
                "size": ("INT", {"default": 36, "min": 0, "max": 255, "step": 1}),
                "font_R": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                "font_G": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                "font_B": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                "font_A": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
                "background_R": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
                "background_G": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
                "background_B": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
                "background_A": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
                "width": ("INT", {"default": 128, "min": 0,  "step": 1}),
                "height": ("INT", {"default": 128, "min": 0,  "step": 1}),
                "expand": (["true", "false"], {"default": "true"}),
                "x": ("INT", {"default": 0, "min": -100, "step": 1}),
                "y": ("INT", {"default": 0, "min": -100, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "create_image_new"
    OUTPUT_NODE = False
    CATEGORY = "O/text"

    def create_image_new(self, text, font, size, font_R, font_G, font_B, font_A, background_R, background_G, background_B, background_A, width, height, expand, x, y):
        font_color = (font_R, font_G, font_B, font_A)
        background_color = (background_R, background_G,
                            background_B, background_A)

        font_path = os.path.join(self.font_filepath, font)
        font = ImageFont.truetype(font_path, size)

        # Initialize the drawing context
        image = Image.new('RGBA', (1, 1), color=background_color)
        draw = ImageDraw.Draw(image)

        # Get the size of the text
        text_width, text_height = draw.textsize(text, font=font)

        # Set the dimensions of the image
        if expand == "true":
            if width < text_width:
                width = text_width
            if height < text_height:
                height = text_height

        width = enforce_mul_of_64(width)
        height = enforce_mul_of_64(height)

        # Create a new image
        image = Image.new('RGBA', (width, height), color=background_color)

        # Initialize the drawing context
        draw = ImageDraw.Draw(image)

        # Calculate the position of the text
        text_x = x - text_width/2
        if (text_x < 0):
            text_x = 0
        if (text_x > width-text_width):
            text_x = width - text_width

        text_y = y - text_height/2
        if (text_y < 0):
            text_y = 0
        if (text_y > height-text_height):
            text_y = height - text_height

        # Draw the text on the image
        draw.text((text_x, text_y), text, fill=font_color, font=font)

        # Convert the PIL Image to a tensor
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).unsqueeze(0)
        return image_tensor, {"ui": {"images": image_tensor}}

# region text/NSP


nspterminology = None  # Cache the NSP terminology


def laodNSP():
    global nspterminology
    if (nspterminology != None):
        return nspterminology
    # Fetch the NSP Pantry
    local_pantry = os.path.join(SUIT_DIR, "nsp_pantry.json")

    if not os.path.exists(local_pantry):
        print(f'{PACKAGE_NAME}:downloading NSP')
        response = urlopen(
            'https://raw.githubusercontent.com/WASasquatch/noodle-soup-prompts/main/nsp_pantry.json')
        tmp_pantry = json.loads(response.read())
        # Dump JSON locally
        pantry_serialized = json.dumps(tmp_pantry, indent=4)
        with open(local_pantry, "w") as f:
            f.write(pantry_serialized)
        del response, tmp_pantry

    # Load local pantry
    with open(local_pantry, 'r') as f:
        nspterminology = json.load(f)

    print(f'{PACKAGE_NAME}:NSP ready')
    return nspterminology


class RandomNSP_O:
    @classmethod
    def laodCategories(s):
        nspterminology = laodNSP()
        terminologies = []
        for term in nspterminology:
            terminologies.append(term)

        return (terminologies)

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "terminology": (s.laodCategories(),),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
        }}
    RETURN_TYPES = ("STRING",)
    FUNCTION = "fun"

    CATEGORY = "O/text/NSP"

    def fun(self, terminology, seed):

        nspterminology = laodNSP()
        # Set the seed
        random.seed(seed)

        result = random.choice(nspterminology[terminology])
        return (result, {"ui": {"STRING": result}})


class ConcatRandomNSP_O:
    @classmethod
    def laodCategories(s):
        nspterminology = laodNSP()
        terminologies = []
        for term in nspterminology:
            terminologies.append(term)

        return (terminologies)

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "text": ("STRING", {"multiline": False, "defaultBehavior": "input"}),
            "terminology": (s.laodCategories(),),
            "separator": ("STRING", {"multiline": False, "default": ","}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
        }}
    RETURN_TYPES = ("STRING",)
    FUNCTION = "fun"
    CATEGORY = "O/text/NSP"

    def fun(self, text, terminology, separator, seed):

        nspterminology = laodNSP()
        # Set the seed
        random.seed(seed)

        result = random.choice(nspterminology[terminology])

        return (text+separator+result+separator, {"ui": {"STRING": result}})
# endregion text/NSP

# region debug text


class DebugText_O:
    """
    This node will write a text to the console
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "text": ("STRING", {"multiline": False, "defaultBehavior": "input"}),
            "prefix": ("STRING", {"default": "debug", "multiline": False}),
        }}

    RETURN_TYPES = ()
    FUNCTION = "debug_string"
    OUTPUT_NODE = True
    CATEGORY = "O/debug/text"

    @staticmethod
    def debug_string(text, prefix):
        print(f'{PACKAGE_NAME}:{prefix}:{text}')
        return ()


class DebugTextRoute_O:
    """
    This node will write a text to the console
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "text": ("STRING", {"multiline": False, "defaultBehavior": "input"}),
            "prefix": ("STRING", {"default": "debug", "multiline": False}),
        }}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "debug_string"
    CATEGORY = "O/debug/text"

    @staticmethod
    def debug_string(text, prefix):
        print(f'{PACKAGE_NAME}:{prefix}:{text}')
        return (text,)


# endregion

# region text/operations


class concat_text_O:
    """
    This node will concatenate two strings together
    """
    @ classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "text1": ("STRING", {"multiline": True, "defaultBehavior": "input"}),
            "text2": ("STRING", {"multiline": True, "defaultBehavior": "input"}),
            "separator": ("STRING", {"multiline": False, "default": ","}),
        }}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "fun"
    CATEGORY = "O/text/operations"

    @ staticmethod
    def fun(text1, separator, text2):
        return (text1 + separator + text2,)


class trim_text_O:
    """
    This node will trim a string from the left and right
    """
    @ classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "text": ("STRING", {"multiline": False, "defaultBehavior": "input"}),
        }}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "fun"
    CATEGORY = "O/text/operations"

    def fun(self, text):
        return (text.strip(),)


class replace_text_O:
    """
    This node will replace a string with another string
    """
    @ classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "text": ("STRING", {"multiline": True, "defaultBehavior": "input"}),
            "old": ("STRING", {"multiline": False}),
            "new": ("STRING", {"multiline": False})
        }}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "fun"
    CATEGORY = "O/text/operations"

    @ staticmethod
    def fun(text, old, new):
        return (text.replace(old, new),)  # replace a text with another text
# endregion
# endregion TextTools

# region Image


def upscaleImage(image, upscale_method, WidthFactor, HeightFactor, crop, MulOf46):
    samples = image.movedim(-1, 1)
    height = HeightFactor * samples.shape[2]
    width = WidthFactor * samples.shape[3]
    if (width > MAX_RESOLUTION):
        width = MAX_RESOLUTION
    if (height > MAX_RESOLUTION):
        height = MAX_RESOLUTION

    if (MulOf46 == "enabled"):
        width = enforce_mul_of_64(width)
        height = enforce_mul_of_64(height)

    width = int(width)
    height = int(height)
    print(
        f'{PACKAGE_NAME}:upscale from ({samples.shape[2]},{samples.shape[3]}) to ({width},{height})')
    s = comfy.utils.common_upscale(
        samples, width, height, upscale_method, crop)
    s = s.movedim(1, -1)
    return (s,)


class ImageScaleFactorSimple_O:
    upscale_methods = ["nearest-exact", "bilinear", "area"]
    crop_methods = ["disabled", "center"]
    toggle = ["enabled", "disabled"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",),
                             "upscale_method": (s.upscale_methods,),
                             "Factor": ("FLOAT", {"default": 1.25, "min": 0.0, "max": 10.0, "step": 0.28125}),
                             "MulOf46": (s.toggle, {"default": "enabled"}),
                             "crop": (s.crop_methods,)
                             }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"

    CATEGORY = "O/image"

    def upscale(self, image, upscale_method, Factor, crop, MulOf46):
        return upscaleImage(image, upscale_method, Factor, Factor, crop, MulOf46)


class ImageScaleFactor_O:
    upscale_methods = ["nearest-exact", "bilinear", "area"]
    crop_methods = ["disabled", "center"]
    toggle = ["enabled", "disabled"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",),
                             "upscale_method": (s.upscale_methods,),
                             "WidthFactor": ("FLOAT", {"default": 1.25, "min": 0.0, "max": 10.0, "step": 0.28125}),
                             "HeightFactor": ("FLOAT", {"default": 1.25, "min": 0.0, "max": 10.0, "step": 0.28125}),
                             "MulOf46": (s.toggle, {"default": "enabled"}),
                             "crop": (s.crop_methods,)
                             }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"

    CATEGORY = "O/image"

    def upscale(self, image, upscale_method, WidthFactor, HeightFactor, crop, MulOf46):
        return upscaleImage(image, upscale_method, WidthFactor, HeightFactor, crop, MulOf46)

# endregion

# region numbers


def solveEquation(equation):
    answer = 0.0
    # Check if v is a valid equation or a number using regular expressions
    try:
        # Solve the equation using Python's built-in eval function
        answer = eval(equation)
    except Exception as e:
        print(f'{PACKAGE_NAME}: equation is not valid: {equation} error: {e}')
        answer = "NAN"

    return answer


class applyEquation1param_O:
    """
    This node generate seeds for the model
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "x": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0xffffffffffffffff, "defaultBehavior": "input"}),
            "equation": ("STRING", {"multiline": True, "default": "x*1"}),
        }
        }

    RETURN_TYPES = ("FLOAT", "int",)
    FUNCTION = "fun"
    CATEGORY = "O/numbers"

    def fun(self, x, equation):
        equation = equation.replace("x", "("+str(x)+")")
        answer = solveEquation(equation)
        return (answer, int(answer), )


class applyEquation2params_O:
    """
    This node generate seeds for the model
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "x": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0xffffffffffffffff, "defaultBehavior": "input"}),
            "y": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0xffffffffffffffff, "defaultBehavior": "input"}),
            "equation": ("STRING", {"multiline": True, "default": "x+y"}),
        },
            "optional": {
            "equation_2": ("STRING", {"multiline": True, "default": "x+y"}),
        }
        }

    RETURN_TYPES = ("FLOAT", "INT", "FLOAT", "INT")
    FUNCTION = "fun"
    CATEGORY = "O/numbers"

    def fun(self, x, y, equation, equation_2):

        answer = 0.0
        answer_2 = 0.0

        if (equation != ""):
            equation = equation.replace("x", "("+str(x)+")")
            equation = equation.replace("y", "("+str(y)+")")
            answer = solveEquation(equation)

        if (equation_2 != ""):
            equation_2 = equation_2.replace("x", "("+str(x)+")")
            equation_2 = equation_2.replace("y", "("+str(y)+")")
            answer_2 = solveEquation(equation_2)

        return (answer, int(answer), answer_2, int(answer_2),)


class floatToInt_O:
    """
    This node convert float to int
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "float": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0xffffffffffffffff, "defaultBehavior": "input"}),
        }
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "fun"
    CATEGORY = "O/numbers"

    def fun(self, float):
        return (int(float),)


class intToFloat_O:
    """
    This node convert int to float
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "int": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "defaultBehavior": "input"}),
        }
        }

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "fun"
    CATEGORY = "O/numbers"

    def fun(self, int):
        return (float(int),)


class floatToText_O:
    """
    This node convert float to text
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "float": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0xffffffffffffffff, "defaultBehavior": "input"}),
        }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "fun"
    CATEGORY = "O/numbers"

    def fun(self, float):
        return (str(float),)


class GetImageWidthAndHeight_O:
    upscale_methods = ["nearest-exact", "bilinear", "area"]
    crop_methods = ["disabled", "center"]
    toggle = ["enabled", "disabled"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",),
                             }
                }
    RETURN_TYPES = ("INT", "INT")
    FUNCTION = "fun"

    CATEGORY = "O/numbers"

    def fun(self, image):
        samples = image.movedim(-1, 1)
        height = samples.shape[2]
        width = samples.shape[3]
        return (int(width), int(height),)


class GetLatentWidthAndHeight_O:
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
            }
        }

    RETURN_TYPES = ("INT", "INT",)
    FUNCTION = "fun"
    CATEGORY = "O/numbers"

    def fun(self, samples):
        w = samples["samples"].shape[3]
        h = samples["samples"].shape[2]
        return (int(w), int(h),)

# endregion

# region Utils


class Text_O:
    """
    to provide text to the model
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "fun"
    CATEGORY = "O/utils"

    def fun(self, text):
        return (text+" ",)


class seed_O:
    """
    This node generate seeds for the model
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}), }}

    RETURN_TYPES = ("INT",)
    FUNCTION = "fun"
    CATEGORY = "O/utils"

    def fun(self, seed):
        return (seed,)


class int_O:
    """
    This node generate seeds for the model
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"int": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}), }}

    RETURN_TYPES = ("INT",)
    FUNCTION = "fun"
    CATEGORY = "O/utils"

    def fun(self, int):
        return (int,)


class float_O:
    """
    This node generate seeds for the model
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"float": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0xffffffffffffffff}), }}

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "fun"
    CATEGORY = "O/utils"

    def fun(self, float):
        return (float,)


class Note_O:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True})}}
    RETURN_TYPES = ()
    FUNCTION = "fun"
    OUTPUT_NODE = True
    CATEGORY = "O/utils"

    def fun(self, text):
        return ()
# endregion


# Define the node class mappings
NODE_CLASS_MAPPINGS = {
    # openAITools------------------------------------------
    "ChatGPT Simple _O": O_ChatGPT_O,
    "ChatGPT compact _O": O_ChatGPT_medium_O,
    # openAiTools > Advanced
    "load_openAI _O": load_openAI_O,
    # openAiTools > Advanced > ChatGPT
    "Chat_Message _O": openAi_chat_message_O,
    "combine_chat_messages _O": openAi_chat_messages_Combine_O,
    "Chat completion _O": openAi_chat_completion_O,
    # openAiTools > Advanced > image
    "create image _O": openAi_Image_create_O,
    # "Edit_image _O": openAi_Image_Edit, # coming soon
    "variation_image _O": openAi_Image_variation_O,
    # latentTools------------------------------------------
    "LatentUpscaleFactor _O": LatentUpscaleFactor_O,
    "LatentUpscaleFactorSimple _O": LatentUpscaleFactorSimple_O,
    "selectLatentFromBatch _O": SelectLatentImage_O,
    # "VAEDecodeParallel _O": VAEDecodeParallel_O, # coming soon
    # StringTools------------------------------------------
    "RandomNSP _O": RandomNSP_O,
    "ConcatRandomNSP_O": ConcatRandomNSP_O,
    "Concat Text _O": concat_text_O,
    "Trim Text _O": trim_text_O,
    "Replace Text _O": replace_text_O,
    "saveTextToFile _O": saveTextToFile_O,
    "Text2Image _O": Text2Image_O,
    # ImageTools------------------------------------------
    "ImageScaleFactor _O": ImageScaleFactor_O,
    "ImageScaleFactorSimple _O": ImageScaleFactorSimple_O,
    # NumberTools------------------------------------------
    "Equation1param _O": applyEquation1param_O,
    "Equation2params _O": applyEquation2params_O,
    "floatToInt _O": floatToInt_O,
    "intToFloat _O": intToFloat_O,
    "floatToText _O": floatToText_O,
    "GetImage_(Width&Height) _O": GetImageWidthAndHeight_O,
    "GetLatent_(Width&Height) _O": GetLatentWidthAndHeight_O,
    # debug------------------------------------------
    "debug messages_O": DebugOpenAIChatMEssages_O,
    "debug Completeion _O": DebugOpenAIChatCompletion_O,
    "Debug Text _O": DebugText_O,
    "Debug Text route _O": DebugTextRoute_O,
    # Utils------------------------------------------
    "Note _O": Note_O,
    "Text _O": Text_O,
    "seed _O": seed_O,
    "int _O": int_O,
    "float _O": float_O,
}
