# Developed by Omar - https://github.com/omar92
# https://civitai.com/user/omar92
# discord: Omar92#3374

import os
import numpy as np
import torch
from PIL import Image, ImageFont

class O_String:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"string": ("STRING", {"multiline": True})}}

    RETURN_TYPES = ("STR",)
    FUNCTION = "ostr"
    CATEGORY = "O/string"

    @staticmethod
    def ostr(string):
        return ({"string": string},)

class concat_String:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "string1": ("STR",),
            "string2": ("STR",)
        }}

    RETURN_TYPES = ("STR",)
    FUNCTION = "fun"
    CATEGORY = "O/string"

    @staticmethod
    def fun(string1, string2):
        return ({"string": string1["string"] + string2["string"]},)

class trim_String:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "string": ("STR",),
        }}

    RETURN_TYPES = ("STR",)
    FUNCTION = "fun"
    CATEGORY = "O/string"

    def fun(self, string):
        return (
            {
                "string": (string["string"].strip()),
            },
        )

class replace_String:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "string": ("STR",),
            "old": ("STRING", {"multiline": False}),
            "new": ("STRING", {"multiline": False})
        }}

    RETURN_TYPES = ("STR",)
    FUNCTION = "fun"
    CATEGORY = "O/string"

    @staticmethod
    def fun(string, old, new):
        return ({"string": string["string"].replace(old, new)},)
    
class replace_String_advanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "string": ("STR",),
            "old": ("STR",),
            "new": ("STR",),
        }}

    RETURN_TYPES = ("STR",)
    FUNCTION = "fun"
    CATEGORY = "O/string"

    @staticmethod
    def fun(string, old, new):
        return ({"string": string["string"].replace(old["string"], new["string"])},)

class DebugString:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"string": ("STR",)}}

    RETURN_TYPES = ()
    FUNCTION = "debug_string"
    OUTPUT_NODE = True
    CATEGORY = "O/string"

    @staticmethod
    def debug_string(string):
        print("debugString:", string["string"])
        return ()

class string2Image:
    def __init__(self):
        self.font_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "fonts")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            "string": ("STR",),
            "font": ("STRING", {"default": "CALIBRI.TTF","multiline": False}),
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
    CATEGORY = "O/string"

    def create_image(self,string,font,size,font_R,font_G,font_B,background_R,background_G,background_B):
        font_color = (font_R , font_G , font_B )
        font = ImageFont.truetype(self.font_filepath+"\\"+font , size)
        mask_image = font.getmask(string["string"], "L")
        image = Image.new("RGB", mask_image.size, (background_R, background_G, background_B))
        image.im.paste(font_color, (0, 0) + mask_image.size, mask_image)  # need to use the inner `img.im.paste` due to `getmask` returning a core

        # Convert the  PIL Image to a tensor
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).unsqueeze(0)
        return (image_tensor,)

class CLIPStringEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
              "string": ("STR",),
              "clip": ("CLIP", )
            }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"

    CATEGORY = "O/string"

    def encode(self,string, clip):
        return ([[clip.encode(string["string"]), {}]], )


NODE_CLASS_MAPPINGS = {
    "String _O": O_String,
    "Debug String _O": DebugString,
    "concat Strings _O": concat_String,
    "trim String _O": trim_String,
    "replace String _O": replace_String,
    "replace String advanced _O": replace_String_advanced,
    "string2Image _O": string2Image,
    "CLIPStringEncode _O": CLIPStringEncode,
}

