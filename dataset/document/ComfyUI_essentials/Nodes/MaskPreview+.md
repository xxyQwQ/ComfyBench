---
tags:
- Preview
---

# 🔧 Mask Preview
## Documentation
- Class name: `MaskPreview+`
- Category: `essentials/mask`
- Output node: `True`

The MaskPreview node is designed for generating a preview image from a given mask. It reshapes and expands the mask to fit the image format, then saves the resulting image with a specified prefix, optionally incorporating additional PNG metadata and a prompt.
## Input types
### Required
- **`mask`**
    - The mask input is the primary data used for generating the preview image. It dictates the shape and content of the output image.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
## Output types
The node doesn't have output types
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class MaskPreview(SaveImage):
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"mask": ("MASK",), },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    FUNCTION = "execute"
    CATEGORY = "essentials/mask"

    def execute(self, mask, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        preview = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
        return self.save_images(preview, filename_prefix, prompt, extra_pnginfo)

```
