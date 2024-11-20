---
tags:
- Preview
---

# ImageAndMaskPreview
## Documentation
- Class name: `ImageAndMaskPreview`
- Category: `KJNodes`
- Output node: `True`

This node is designed to generate a preview of an image with an optional mask applied. It supports adjusting the mask's opacity and color, and can handle cases where either the image, the mask, or both are provided. The node also offers functionality to pass through the generated preview without saving, or to save the preview with a specified filename prefix, incorporating additional PNG metadata if provided.
## Input types
### Required
- **`mask_opacity`**
    - Specifies the opacity level of the mask when both an image and a mask are provided, affecting the visibility of the mask overlay on the image.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`mask_color`**
    - Defines the color of the mask in either RGB or hexadecimal format, which is applied when both an image and a mask are present, influencing the appearance of the mask overlay.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`pass_through`**
    - A boolean flag that determines whether the generated preview is directly returned without being saved, allowing for immediate use or further processing.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
- **`image`**
    - The image to be previewed, which can be optionally accompanied by a mask to create a composite preview.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`mask`**
    - The mask to be applied over the image, which can be adjusted in terms of opacity and color, or used alone to generate a mask-only preview.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
## Output types
- **`composite`**
    - Comfy dtype: `IMAGE`
    - The generated preview of the image with the optional mask applied, which can be either passed through directly or saved to a file.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ImageAndMaskPreview(SaveImage):
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask_opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mask_color": ("STRING", {"default": "255, 255, 255"}),
                "pass_through": ("BOOLEAN", {"default": False}),
             },
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK",),                
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("composite",)
    FUNCTION = "execute"
    CATEGORY = "KJNodes"
    DESCRIPTION = """
Preview an image or a mask, when both inputs are used  
composites the mask on top of the image.
with pass_through on the preview is disabled and the  
composite is returned from the composite slot instead,  
this allows for the preview to be passed for video combine  
nodes for example.
"""

    def execute(self, mask_opacity, mask_color, pass_through, filename_prefix="ComfyUI", image=None, mask=None, prompt=None, extra_pnginfo=None):
        if mask is not None and image is None:
            preview = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
        elif mask is None and image is not None:
            preview = image
        elif mask is not None and image is not None:
            mask_adjusted = mask * mask_opacity
            mask_image = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3).clone()

            if ',' in mask_color:
                color_list = np.clip([int(channel) for channel in mask_color.split(',')], 0, 255) # RGB format
            else:
                mask_color = mask_color.lstrip('#')
                color_list = [int(mask_color[i:i+2], 16) for i in (0, 2, 4)] # Hex format
            mask_image[:, :, :, 0] = color_list[0] / 255 # Red channel
            mask_image[:, :, :, 1] = color_list[1] / 255 # Green channel
            mask_image[:, :, :, 2] = color_list[2] / 255 # Blue channel
            
            preview, = ImageCompositeMasked.composite(self, image, mask_image, 0, 0, True, mask_adjusted)
        if pass_through:
            return (preview, )
        return(self.save_images(preview, filename_prefix, prompt, extra_pnginfo))

```
