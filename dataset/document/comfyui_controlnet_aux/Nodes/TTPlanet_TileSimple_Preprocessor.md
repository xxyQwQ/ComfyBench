---
tags:
- Image
- Tiled
---

# TTPlanet Tile Simple
## Documentation
- Class name: `TTPlanet_TileSimple_Preprocessor`
- Category: `ControlNet Preprocessors/tile`
- Output node: `False`

This node is designed for preprocessing images by applying a simple tiling effect, which involves scaling and blurring to enhance or modify the image's appearance for further processing or analysis.
## Input types
### Required
- **`image`**
    - The input image to be processed. It serves as the primary data for the tiling effect application.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`scale_factor`**
    - Determines the scaling factor for the image, affecting the size of the tiles in the processed image.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`blur_strength`**
    - Adjusts the strength of the blur applied to the image, influencing the smoothness of the tile edges.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The processed image with the applied tiling effect, ready for further processing or analysis.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class TTPlanet_TileSimple_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "scale_factor": ("FLOAT", {"default": 1.00, "min": 1.00, "max": 8.00, "step": 0.05}),
                "blur_strength": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 10.0, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors/tile"

    def execute(self, image, scale_factor, blur_strength):
        from controlnet_aux.tile import TTPLanet_Tile_Detector_Simple

        return (common_annotator_call(TTPLanet_Tile_Detector_Simple(), image, scale_factor=scale_factor, blur_strength=blur_strength),)

```
