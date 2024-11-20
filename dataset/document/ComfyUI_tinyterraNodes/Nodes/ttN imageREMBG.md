---
tags:
- BackgroundRemoval
- Image
---

# imageRemBG
## Documentation
- Class name: `ttN imageREMBG`
- Category: `🌏 tinyterra/image`
- Output node: `True`

The node is designed to remove the background from images using the REMBG library, converting images to a format suitable for further processing or visualization. It emphasizes the extraction of the main subject by eliminating the background, potentially enhancing the focus on the subject matter for subsequent operations.
## Input types
### Required
- **`image`**
    - The input image to be processed for background removal. This parameter is crucial as it directly influences the effectiveness of the background removal process.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`image_output`**
    - Determines how the output image is handled, offering options such as hiding, previewing, saving, or a combination of hiding and saving. This parameter affects the output format and visibility of the processed image.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`save_prefix`**
    - Specifies the prefix for saving the processed image, indicating how the output file names should begin. This parameter is essential when the image output is set to save, guiding the naming convention of the saved files.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The processed image tensor with the background removed, ready for further use.
    - Python dtype: `torch.Tensor`
- **`mask`**
    - Comfy dtype: `MASK`
    - A mask tensor indicating the areas of the image where the background was removed.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ttN_imageREMBG:
    version = '1.0.0'
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "image": ("IMAGE",),
                "image_output": (["Hide", "Preview", "Save", "Hide/Save"],{"default": "Preview"}),
                               "save_prefix": ("STRING", {"default": "ComfyUI"}),
                },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID",
                           "ttNnodeVersion": ttN_imageREMBG.version},
            }


    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "remove_background"
    CATEGORY = "🌏 tinyterra/image"
    OUTPUT_NODE = True

    def remove_background(self, image, image_output, save_prefix, prompt, extra_pnginfo, my_unique_id):
        try:
            from rembg import remove
        except ImportError:
            raise ImportError("REMBG is not installed.\nPlease install it with `pip install rembg` or from https://github.com/danielgatis/rembg.")
        
        image = remove(ttNsampler.tensor2pil(image))
        tensor = ttNsampler.pil2tensor(image)

        #Get alpha mask
        if image.getbands() != ("R", "G", "B", "A"):
            image = image.convert("RGBA")
        mask = None
        if "A" in image.getbands():
            mask = np.array(image.getchannel("A")).astype(np.float32) / 255.0
            mask = torch.from_numpy(mask)
            mask = 1. - mask
        else:
            mask = torch.zeros((64,64), dtype=torch.float32, device=sampler.device)

        if image_output == "Disabled":
            results = []
        else:
            ttN_save = ttNsave(my_unique_id, prompt, extra_pnginfo)
            results = ttN_save.images(tensor, save_prefix, image_output)

        if image_output in ("Hide", "Hide/Save"):
            return (tensor, mask)

        # Output image results to ui and node outputs
        return {"ui": {"images": results},
                "result": (tensor, mask)}

```
