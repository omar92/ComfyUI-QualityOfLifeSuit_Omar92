# ComfyUI-extra-nodes - quality of life
Extra nodes to be used in ComfyUI, including a new ChatGPT node for generating natural language responses.

## ComfyUI
ComfyUI is an advanced node-based UI that utilizes Stable Diffusion, allowing you to create customized workflows such as image post-processing or conversions.

## How to install
Download the zip file.
Extract to ..\ComfyUI\custom_nodes.
Restart ComfyUI if it was running (reloading the web is not enough).
You will find my nodes under the new group O/....

## How to update
- quality of life will auto update each time you run comfyUI
- if you want to stop autoUpdate edit  __init__.py  and set autoUpdate = false

## Current nodes
## ChatGPT
This node harnesses the power of chatGPT, an advanced language model that can generate detailed image descriptions from a small input.
- you need to have  OpenAI API key , which you can find at https://beta.openai.com/docs/developer-apis/overview
- Once you have your API key, add it to the api_key.txt file
- I have made it a separate file, so that the API key doesn't get embedded in the generated images.

## String Suit
This set of nodes adds support for string manipulation and includes a tool to generate an image from text.

- String: A node that can hold a string (text).
- Debug String: This node writes the string to the console.
- Concat String: This node combines two strings together.
- Trim String: This node removes any extra spaces at the start or end of a string.
- Replace String & Replace String Advanced: These nodes replace part of the text with another part.
### String2image
This node generates an image based on text, which can be used with ControlNet to add text to the image. The tool supports various fonts; you can add the font you want in the fonts folder. If you load the example image in ComfyUI, the workflow that generated it will be loaded.

### CLIPStringEncode
This node is a variant of the ClipTextEncode node, but it receives the text from the string node, so you don't have to retype your prompt twice anymore.

## Other tools
### LatentUpscaleMultiply
This node is a variant of the original LatentUpscale tool, but instead of using width and height, you use a multiply number. For example, if the original image dimensions are (512,512) and the mul values are (2,2), the result image will be (1024,1024). You can also use it to downscale by using fractions, e.g., (512,512) mul (.5,.5) → (256,256).

Node Path: O/Latent/LatentUpscaleMultiply

## Thanks for reading my message, and I hope that my tools will help you.

## Contact
### Discord: Omar92#3374
### GitHub: omar92 (https://github.com/omar92)
