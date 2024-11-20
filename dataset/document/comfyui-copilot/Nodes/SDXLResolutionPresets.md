# SDXL Resolution Presets (ws)
## Documentation
- Class name: `SDXLResolutionPresets`
- Category: `copilot`
- Output node: `False`

The SDXLResolutionPresets node provides a selection of predefined image resolution presets for use in image generation tasks, allowing users to easily choose from a variety of common aspect ratios and dimensions.
## Input types
### Required
- **`resolution`**
    - Specifies the desired image resolution from a predefined list of common resolutions, influencing the output image's dimensions.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`width`**
    - Comfy dtype: `INT`
    - The width component of the selected resolution preset.
    - Python dtype: `int`
- **`height`**
    - Comfy dtype: `INT`
    - The height component of the selected resolution preset.
    - Python dtype: `int`
## Usage tips
- Infra type: `CPU`
- Common nodes:
    - [SVD_img2vid_Conditioning](../../Comfy/Nodes/SVD_img2vid_Conditioning.md)
    - [Eff. Loader SDXL](../../efficiency-nodes-comfyui/Nodes/Eff. Loader SDXL.md)
    - [EmptyLatentImage](../../Comfy/Nodes/EmptyLatentImage.md)
    - LatentGaussianNoise
    - [CM_IntToNumber](../../ComfyMath/Nodes/CM_IntToNumber.md)



## Source code
```python
class SDXLResolutionPresets(BaseNode):
    RESOLUTIONS: list[TResolution] = [
        "Square (1024x1024)",
        "Cinematic (1536x640)",
        "Cinematic (640x1536)",
        "Widescreen (1344x768)",
        "Widescreen (768x1344)",
        "Photo (1216x832)",
        "Photo (832x1216)",
        "Portrait (1152x896)",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "resolution": (cls.RESOLUTIONS, {"default": "Square (1024x1024)"}),
            },
        }

    RETURN_TYPES = ("INT", "INT",)
    RETURN_NAMES = ("width", "height",)
    FUNCTION = "get_value"

    def get_value(self, resolution: TResolution, ) -> tuple[int, int]:
        if resolution == "Cinematic (1536x640)":
            return 1536, 640
        if resolution == "Cinematic (640x1536)":
            return 640, 1536
        if resolution == "Widescreen (1344x768)":
            return 1344, 768
        if resolution == "Widescreen (768x1344)":
            return 768, 1344
        if resolution == "Photo (1216x832)":
            return 1216, 832
        if resolution == "Photo (832x1216)":
            return 832, 1216
        if resolution == "Portrait (1152x896)":
            return 1152, 896
        if resolution == "Portrait (896x1152)":
            return 896, 1152
        if resolution == "Square (1024x1024)":
            return 1024, 1024

```
