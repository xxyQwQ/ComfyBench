---
tags:
- Image
---

# Stability API SD3
## Documentation
- Class name: `StabilityAPI_SD3`
- Category: `KJNodes/experimental`
- Output node: `False`

The StabilityAPI_SD3 node interfaces with the StabilityAI API to facilitate image generation or manipulation tasks. It leverages the StabilityAI platform's capabilities, allowing users to generate images based on textual descriptions or modify existing images. This node emphasizes the use of specific API keys for operation and warns against potential metadata storage issues when saving images through separate nodes.
## Input types
### Required
- **`prompt`**
    - The main textual description guiding the image generation or manipulation process, serving as the primary input for creating or altering images.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`n_prompt`**
    - A negative textual description to guide the image generation process by specifying what to avoid, refining the output.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`seed`**
    - A seed value for ensuring reproducibility of the generated images.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`model`**
    - Specifies the model used by StabilityAI for the image generation or manipulation task.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`aspect_ratio`**
    - The desired aspect ratio for the output image, influencing its dimensions.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `float`
- **`output_format`**
    - The format in which the generated image will be outputted, such as PNG or JPEG.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
### Optional
- **`api_key`**
    - The API key required for accessing the StabilityAI platform's services.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`image`**
    - An existing image to be modified or used as a basis for generation in image-to-image tasks.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`img2img_strength`**
    - Controls the influence of the input image on the generated output in image-to-image tasks.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`disable_metadata`**
    - A flag to disable metadata storage, addressing privacy or data storage concerns.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The generated or manipulated image as a result of the process, aligning with the provided descriptions and parameters.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class StabilityAPI_SD3:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "n_prompt": ("STRING", {"multiline": True}),
                "seed": ("INT", {"default": 123,"min": 0, "max": 4294967294, "step": 1}),
                "model": (
                [   
                    'sd3',
                    'sd3-turbo',
                ],
                {
                "default": 'sd3'
                 }),
                 "aspect_ratio": (
                [   
                    '1:1',
                    '16:9',
                    '21:9',
                    '2:3',
                    '3:2',
                    '4:5',
                    '5:4',
                    '9:16',
                    '9:21',
                ],
                {
                "default": '1:1'
                }),
                "output_format": (
                [   
                    'png',
                    'jpeg',
                ],
                {
                "default": 'jpeg'
                 }),                 
            },
            "optional": {
                "api_key": ("STRING", {"multiline": True}),
                "image": ("IMAGE",),
                "img2img_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "disable_metadata": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apicall"

    CATEGORY = "KJNodes/experimental"
    DESCRIPTION = """
## Calls StabilityAI API
   
Although you may have multiple keys in your account,  
you should use the same key for all requests to this API.  

Get your API key here: https://platform.stability.ai/account/keys  
Recommended to set the key in the config.json -file under this  
node packs folder.  
# WARNING:  
Otherwise the API key may get saved in the image metadata even  
with "disable_metadata" on if the workflow includes save nodes  
separate from this node.  
   
sd3 requires 6.5 credits per generation  
sd3-turbo requires 4 credits per generation  

If no image is provided, mode is set to text-to-image  

"""

    def apicall(self, prompt, n_prompt, model, seed, aspect_ratio, output_format, 
                img2img_strength=0.5, image=None, disable_metadata=True, api_key=""):
        from comfy.cli_args import args
        if disable_metadata:
            args.disable_metadata = True
        else:
            args.disable_metadata = False
        
        import requests
        from torchvision import transforms
        
        data = {
                "mode": "text-to-image",
                "prompt": prompt,
                "model": model,
                "seed": seed,
                "output_format": output_format
                }
        
        if image is not None:
            image = image.permute(0, 3, 1, 2).squeeze(0)
            to_pil = transforms.ToPILImage()
            pil_image = to_pil(image)
            # Save the PIL Image to a BytesIO object
            buffer = io.BytesIO()
            pil_image.save(buffer, format='PNG')
            buffer.seek(0)
            files = {"image": ("image.png", buffer, "image/png")}
           
            data["mode"] = "image-to-image"
            data["image"] = pil_image
            data["strength"] = img2img_strength
        else:
            data["aspect_ratio"] = aspect_ratio,
            files = {"none": ''}
        
        if model != "sd3-turbo":
            data["negative_prompt"] = n_prompt

        headers={
                "accept": "image/*"
            }
        
        if api_key != "":
            headers["authorization"] = api_key
        else:
            config_file_path = os.path.join(script_directory,"config.json")
            with open(config_file_path, 'r') as file:
                config = json.load(file)
            api_key_from_config = config.get("sai_api_key")
            headers["authorization"] = api_key_from_config            
        
        response = requests.post(
            f"https://api.stability.ai/v2beta/stable-image/generate/sd3",
            headers=headers,
            files = files,
            data = data,
        )

        if response.status_code == 200:
            # Convert the response content to a PIL Image
            image = Image.open(io.BytesIO(response.content))
            # Convert the PIL Image to a PyTorch tensor
            transform = transforms.ToTensor()
            tensor_image = transform(image)
            tensor_image = tensor_image.unsqueeze(0)
            tensor_image = tensor_image.permute(0, 2, 3, 1).cpu().float()
            return (tensor_image,)
        else:
            try:
                # Attempt to parse the response as JSON
                error_data = response.json()
                raise Exception(f"Server error: {error_data}")
            except json.JSONDecodeError:
                # If the response is not valid JSON, raise a different exception
                raise Exception(f"Server error: {response.text}")

```
