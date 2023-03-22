# Developed by Omar - https://github.com/omar92
# https://civitai.com/user/omar92
# discord: Omar92#3374

# This node provides an alterntive scaling node buy multiplying previous width and height by a factor
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))
import comfy.samplers
import comfy.sd
import comfy.utils
import model_management
def before_node_execution():
    model_management.throw_exception_if_processing_interrupted()
def interrupt_processing(value=True):
    model_management.interrupt_current_processing(value)

class LatentUpscaleMultiply:
    upscale_methods = ["nearest-exact", "bilinear", "area"]
    crop_methods = ["disabled", "center"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT",),
                "upscale_method": (s.upscale_methods,),
                "WidthMul": (
                    "FLOAT",
                    {"default": 1.25, "min": 0.0, "max": 10.0, "step": 0.1},
                ),
                "HeightMul": (
                    "FLOAT",
                    {"default": 1.25, "min": 0.0, "max": 10.0, "step": 0.1},
                ),
                "crop": (s.crop_methods,),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "upscale"

    CATEGORY = "O/latent"

    def upscale(self, samples, upscale_method, WidthMul, HeightMul, crop):
        _s = samples.copy()
        _x = samples["samples"].shape[3]
        _y = samples["samples"].shape[2]

        new_x = int(_x * WidthMul)
        new_y = int(_y * HeightMul)
        print("upscale from ("+ str(_x * 8)+ ","+ str(_y * 8)+ ") to ("+ str(new_x * 8)+ ","+ str(new_y * 8)+ ")")

        def enforce_mul_of_64(d):
            leftover = d % 8
            if leftover != 0:
                d += 8 - leftover
            return d

        _s["samples"] = comfy.utils.common_upscale(
            samples["samples"], enforce_mul_of_64(new_x), enforce_mul_of_64(new_y), upscale_method, crop
        )
        return (_s,)


NODE_CLASS_MAPPINGS = {"LatentUpscaleMultiply": LatentUpscaleMultiply}
