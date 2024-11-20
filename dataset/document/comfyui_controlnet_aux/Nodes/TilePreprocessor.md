---
tags:
- Image
- Tiled
---

# Tile
## Documentation
- Class name: `TilePreprocessor`
- Category: `ControlNet Preprocessors/tile`
- Output node: `False`

The TilePreprocessor node is designed for preprocessing images to enhance or modify their features for better analysis or further processing. It specifically focuses on manipulating the image's resolution and structure through pyramid upscaling iterations, making it suitable for tasks that require refined image details.
## Input types
### Required
- **`image`**
    - The input image to be processed. It serves as the primary data for the node's operation, influencing the outcome of the preprocessing.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
### Optional
- **`pyrUp_iters`**
    - Specifies the number of pyramid upscaling iterations to apply to the image. This parameter directly affects the level of detail enhancement in the processed image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`resolution`**
    - Defines the target resolution for the image after processing. This parameter helps in adjusting the image's size and detail level to suit specific requirements.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The processed image after pyramid upscaling iterations. It reflects the modifications made to the original image, with potentially enhanced detail and resolution.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - [ControlNetApplyAdvanced](../../Comfy/Nodes/ControlNetApplyAdvanced.md)
    - [CR Thumbnail Preview](../../ComfyUI_Comfyroll_CustomNodes/Nodes/CR Thumbnail Preview.md)
    - [PreviewImage](../../Comfy/Nodes/PreviewImage.md)
    - [ACN_AdvancedControlNetApply](../../ComfyUI-Advanced-ControlNet/Nodes/ACN_AdvancedControlNetApply.md)



## Source code
```python
class Tile_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return create_node_input_types(
            pyrUp_iters = ("INT", {"default": 3, "min": 1, "max": 10, "step": 1})
        )
        

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors/tile"

    def execute(self, image, pyrUp_iters, resolution=512, **kwargs):
        from controlnet_aux.tile import TileDetector

        return (common_annotator_call(TileDetector(), image, pyrUp_iters=pyrUp_iters, resolution=resolution),)

```
