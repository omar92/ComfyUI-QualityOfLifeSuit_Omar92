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
- when you run comfyUI, the suit will generate a config file
The file looks like this :

{
"autoUpdate": true,
"branch": "main",
"openAI_API_Key": "sk-#################################"
}

- if you want to stop autoUpdate edit  `config.json` set "autoUpdate": false

## Current nodes
## openAI suite

## ChatGPT simple
This node harnesses the power of chatGPT, an advanced language model that can generate detailed image descriptions from a small input.
- you need to have  OpenAI API key , which you can find at https://beta.openai.com/docs/developer-apis/overview
- Once you have your API key, add it to the `config.json` file
- I have made it a separate file, so that the API key doesn't get embedded in the generated images.

## advanced openAI
- load_openAI:load openAI module 
### ChatGPT
- Chat_Message: creates a message to be sent to chatGPT
- combine_chat_messages : combine 2 messages together 
- Chat completion: send the messages to ChatGPT and receive answer
### DalE-2
- create image
- variation_image

## String Suit
This set of nodes adds support for string manipulation and includes a tool to generate an image from text.

- Concat String: This node combines two strings together.
- Trim String: This node removes any extra spaces at the start or end of a string.
- Replace String : This nodes replace part of the text with another part.
- Debug String: This node writes the string to the console.
- Debug String route: This node writes the string to the console but will output the same string so that you can add it in middle of a route.
### String2image
This node generates an image based on text, which can be used with ControlNet to add text to the image. The tool supports various fonts; you can add the font you want in the fonts folder. If you load the example image in ComfyUI, the workflow that generated it will be loaded.

### save text
- saveTextToFile: this node will save input text to a file "the file will be generated inside  /output folder"
### NSP
"node soup" which is a collection of different values categorized under different terminologies that you can use to generate new prompts easily 
- RandomNSP: returns a random value from the selected terminology 
- ConcatRandomNSP: will append a random value from the selected terminology to the input text (can be used mid route)

## latentTools
### selectLatentFromBatch
this node allow you to select 1 latent image from image batch 
for example if you generate 4 images, it allow you to select 1 of them to do further processing on it 
or you can use it to process them sequentially 
### LatentUpscaleFactor & LatentUpscaleFactorSimple 
This node is a variant of the original LatentUpscale tool, but instead of using width and height, you use a multiply number. For example, if the original image dimensions are (512,512) and the mul values are (2,2), the result image will be (1024,1024). You can also use it to downscale by using fractions, e.g., (512,512) mul (.5,.5) → (256,256).

## ImageTools
### ImageScaleFactor & ImageScaleFactorSimple  
This node is a variant of the original LatentUpscale tool, but instead of using width and height, you use a multiply number. For example, if the original image dimensions are (512,512) and the mul values are (2,2), the result image will be (1024,1024). You can also use it to downscale by using fractions, e.g., (512,512) mul (.5,.5) → (256,256).


## Thanks for reading my message, and I hope that my tools will help you.

## Contact
### Discord: Omar92#3374
### GitHub: omar92 (https://github.com/omar92)
