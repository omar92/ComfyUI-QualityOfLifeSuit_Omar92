# ComfyUI-extra-nodes---quality-of-life
extra nodes to be used in comfyUI

# ComfyUI:
ComfyUI is an advanced node based UI utilizing Stable Diffusion. It allows you to create customized workflows such as image post-processing, or conversions.

# How to install:

Download the zip file.

Extract to ..\ComfyUI\custom_nodes

restart comfy if it was running (reload web, not enough)

you will find my nodes under new group O/…

# Current Nodes:
## String Suit

add multiple nodes to support string manipulation also a tool to generate image from text

### String:
node that can hold string (text)

### Debug String
this node will write the string on the console

### Concat string
this node is used to combine two strings together

### Trim string
this is used to remove any extra spaces at the start or the end of a string

### Replace string & replace string advanced
used to replace part of the text by another part

>>>> String2image <<<<
this node will generate an images based on a text, which can be used with controlNet to add text to the image.
— the tool support fonts “add the font you want in fonts folder”
“If you load the example image in comfyUI the workflow that generated it will be loaded”

>>>>CLIPStringEncode <<< 
The normal ClipTextEncode node but this one receive the text from the string node, so you don't have to retype your prompt twice anymore

## Other tools

## LatentUpscaleMultiply:
it is a variant from the original LatentUpscale tool but instead of using width and height you use a multiply number
for example, if the original images dimensions are (512,512) and the mul values were (2,2) the result image will be (1024,1024)
also you can use it to downscale if needed by using fractions ex:(512,512) mul (.5,.5) → (256,256)
Node Path: O/Latent/LatentUpscaleMultiply

thanks for reading my message, I hope that my tools will help you.

### Discord: Omar92#3374
### Githup: omar92 (https://github.com/omar92)
