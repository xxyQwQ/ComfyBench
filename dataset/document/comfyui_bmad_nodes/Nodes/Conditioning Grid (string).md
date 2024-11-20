---
tags:
- Conditioning
---

# Conditioning Grid (string)
## Documentation
- Class name: `Conditioning Grid (string)`
- Category: `Bmad/conditioning`
- Output node: `False`

This node automates the process of generating conditionings for a grid layout by encoding text inputs using ClipTextEncode and then applying these conditionings to the grid's AreaConditioners. It simplifies the creation of complex conditioning structures for grid-based layouts, making it easier to customize and generate content based on textual descriptions.
## Input types
### Required
- **`clip`**
    - The CLIP model used for encoding the text inputs. It's crucial for converting textual descriptions into a format that can be utilized for conditioning the grid.
    - Comfy dtype: `CLIP`
    - Python dtype: `object`
- **`base`**
    - A base text input that serves as the foundational conditioning for the entire grid. It's encoded and applied to set a baseline for further customization.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`columns`**
    - Specifies the number of columns in the grid, determining its horizontal structure.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`rows`**
    - Defines the number of rows in the grid, shaping its vertical layout.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`width`**
    - The width of each cell in the grid, affecting the resolution and detail of the conditioned content.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - The height of each cell in the grid, influencing the vertical resolution and detail of the conditioned content.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`strength`**
    - Controls the intensity of the applied conditioning, allowing for fine-tuning of the grid's overall effect.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`conditioning`**
    - Comfy dtype: `CONDITIONING`
    - The resulting conditioning structure for the grid, ready to be used for generating content based on the encoded text inputs.
    - Python dtype: `object`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ConditioningGridStr:
    """
    Node similar to ConditioningGridCond, but automates an additional step, using a ClipTextEncode per text input.
    Each conditioning obtained from the text inputs is then used as input for the Grid's AreaConditioners.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "clip": ("CLIP",),
            "base": ("STRING", {"default": '', "multiline": False}),
            "columns": grid_len_INPUT,
            "rows": grid_len_INPUT,
            "width": ("INT", {"default": 256, "min": 16, "max": 2048, "step": 1}),
            "height": ("INT", {"default": 256, "min": 16, "max": 2048, "step": 1}),
            "strength": ("FLOAT", {"default": 3, }),
        }}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "set_conditioning"
    CATEGORY = conditioning_category_path

    def set_conditioning(self, clip, base, columns, rows, width, height, strength, **kwargs):
        text_encode_node = nodes.CLIPTextEncode()
        cond_grid_node = ConditioningGridCond()

        encoded_base = text_encode_node.encode(clip, base)[0]
        encoded_grid = {}
        for r in range(rows):
            for c in range(columns):
                cell = f"r{r + 1}_c{c + 1}"
                encoded_grid[cell] = text_encode_node.encode(clip, kwargs[cell])[0]

        return cond_grid_node.set_conditioning(encoded_base, columns, rows, width, height, strength, **encoded_grid)

```
