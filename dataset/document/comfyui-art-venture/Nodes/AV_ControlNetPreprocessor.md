---
tags:
- DepthMap
- Image
- ImagePreprocessing
---

# ControlNet Preprocessor
## Documentation
- Class name: `AV_ControlNetPreprocessor`
- Category: `Art Venture/Loaders`
- Output node: `False`

The AV_ControlNetPreprocessor node is designed for preparing images for further processing or analysis within the Art Venture framework, specifically by applying various preprocessing techniques to enhance or modify the images based on the selected preprocessor and resolution settings. It supports a range of preprocessing options, including segmentation, edge detection, and artistic style adjustments, tailored to the requirements of subsequent control net detection and application.
## Input types
### Required
- **`image`**
    - The input image to be preprocessed. This is the primary data upon which preprocessing operations are performed, setting the stage for enhanced image analysis or manipulation.
    - Comfy dtype: `IMAGE`
    - Python dtype: `IMAGE`
- **`preprocessor`**
    - Specifies the preprocessing technique to be applied to the input image. This selection determines how the image will be modified or enhanced, impacting the effectiveness of subsequent control net detection.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`sd_version`**
    - Indicates the version of the Stable Diffusion model to be used, affecting the compatibility and performance of the preprocessing and control net detection processes.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
### Optional
- **`resolution`**
    - The target resolution for the preprocessed image. Adjusting this parameter can influence the detail level and quality of the preprocessing output.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`preprocessor_override`**
    - Allows for overriding the default preprocessor choice, providing flexibility in experimenting with different preprocessing techniques.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`IMAGE`**
    - Comfy dtype: `IMAGE`
    - The preprocessed image, ready for further analysis or processing within the Art Venture framework.
    - Python dtype: `IMAGE`
- **`CNET_NAME`**
    - Comfy dtype: `STRING`
    - The name of the detected control net, indicating the specific preprocessing technique applied to the input image.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes:
    - [CR Multi-ControlNet Stack](../../ComfyUI_Comfyroll_CustomNodes/Nodes/CR Multi-ControlNet Stack.md)



## Source code
```python
class AV_ControlNetPreprocessor:
    preprocessors = list(control_net_preprocessors.keys())

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "preprocessor": (["None", "tile"] + s.preprocessors,),
                "sd_version": (["sd15", "sdxl", "sdxl_t2i"],),
            },
            "optional": {
                "resolution": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                "preprocessor_override": ("STRING", {"default": "None"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("IMAGE", "CNET_NAME")
    FUNCTION = "detect_controlnet"
    CATEGORY = "Art Venture/Loaders"

    def detect_controlnet(self, image, preprocessor, sd_version, resolution=512, preprocessor_override="None"):
        if preprocessor_override != "None":
            if preprocessor_override not in control_net_preprocessors:
                print(
                    f"Warning: Not found ControlNet preprocessor {preprocessor_override}. Use {preprocessor} instead."
                )
            else:
                preprocessor = preprocessor_override

        image = apply_preprocessor(image, preprocessor, resolution=resolution)
        control_net_name = detect_controlnet(preprocessor, sd_version)

        return (image, control_net_name)

```
