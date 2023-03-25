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
    CATEGORY = "O/latent"

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
            samples["samples"], enforce_mul_of_64(new_x), enforce_mul_of_64(new_y), upscale_method, crop
        )
        return (s,)


NODE_CLASS_MAPPINGS = {"LatentUpscaleMultiply": LatentUpscaleMultiply}
