---
tags:
- DepthMap
- Image
- ImagePreprocessing
---

# TTPlanet Tile GuidedFilter
## Documentation
- Class name: `TTPlanet_TileGF_Preprocessor`
- Category: `ControlNet Preprocessors/tile`
- Output node: `False`

This node preprocesses images for tile detection by applying a guided filter technique, enhancing the image's features relevant for tile detection. It adjusts the image based on scale factor, blur strength, radius, and epsilon values to prepare it for further processing or analysis.
## Input types
### Required
- **`image`**
    - The input image to be processed. It is the primary data on which the guided filter and other preprocessing steps are applied.
    - Comfy dtype: `IMAGE`
    - Python dtype: `Image`
- **`scale_factor`**
    - Determines the scaling of the image, affecting its resolution and the granularity of details to be preserved or enhanced.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`blur_strength`**
    - Controls the intensity of the blur applied to the image, which can help in reducing noise and improving the effectiveness of the guided filter.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`radius`**
    - Specifies the radius of the guided filter, influencing the size of the local region around each pixel considered during filtering.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`eps`**
    - Epsilon value for the guided filter, determining the degree of smoothing and edge preservation.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The processed image, enhanced and ready for further tile detection steps.
    - Python dtype: `Image`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class TTPlanet_TileGF_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "scale_factor": ("FLOAT", {"default": 1.00, "min": 1.00, "max": 8.00, "step": 0.05}),
                "blur_strength": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                "radius": ("INT", {"default": 7, "min": 1, "max": 20, "step": 1}),
                "eps": ("FLOAT", {"default": 0.01, "min": 0.001, "max": 0.1, "step": 0.001}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors/tile"

    def execute(self, image, scale_factor, blur_strength, radius, eps, **kwargs):
        from controlnet_aux.tile import TTPlanet_Tile_Detector_GF

        return (common_annotator_call(TTPlanet_Tile_Detector_GF(), image, scale_factor=scale_factor, blur_strength=blur_strength, radius=radius, eps=eps),)

```
