---
tags:
- Mask
---

# LayerMask: MaskPreview
## Documentation
- Class name: `LayerMask: MaskPreview`
- Category: `😺dzNodes/LayerMask`
- Output node: `True`

The MaskPreview node is designed to generate a visual preview of a mask by transforming it into an image format suitable for visualization. It extends the functionality of saving images by specifically handling mask data, ensuring that masks can be easily reviewed and verified within a workflow.
## Input types
### Required
- **`mask`**
    - The 'mask' input is crucial for generating a preview of the mask. It represents the mask data that will be transformed into a visual format. The processing of this input directly influences the appearance of the generated preview, making it essential for visual verification of mask conditions.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
## Output types
The node doesn't have output types
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class MaskPreview(SaveImage):
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz1234567890") for x in range(5))
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {"mask": ("MASK",), },
        }

    FUNCTION = "mask_preview"
    CATEGORY = '😺dzNodes/LayerMask'
    OUTPUT_NODE = True

    def mask_preview(self, mask):
        if mask.dim() == 2:
            mask = torch.unsqueeze(mask, 0)
        preview = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
        return self.save_images(preview, "MaskPreview")

```
