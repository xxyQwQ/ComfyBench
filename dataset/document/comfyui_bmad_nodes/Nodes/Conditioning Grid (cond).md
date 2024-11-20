---
tags:
- Conditioning
---

# Conditioning Grid (cond)
## Documentation
- Class name: `Conditioning Grid (cond)`
- Category: `Bmad/conditioning`
- Output node: `False`

This node is designed to apply conditioning to a grid layout, where each cell within the grid can be individually conditioned based on text inputs. It automates the process of encoding text inputs using a CLIP model and then applying these encoded conditionings to specific areas within a grid, facilitating the generation of complex, grid-based visual layouts with varied content.
## Input types
### Required
- **`columns`**
    - Specifies the number of columns in the grid, determining the grid's horizontal layout and how many cells it contains.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`rows`**
    - Specifies the number of rows in the grid, determining the grid's vertical layout and how many cells it contains.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`width`**
    - The width of each cell in the grid, in pixels. This affects the resolution and aspect ratio of the content within each cell.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - The height of each cell in the grid, in pixels. This affects the resolution and aspect ratio of the content within each cell.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`strength`**
    - Controls the intensity of the applied conditioning, allowing for fine-tuning of how prominently the text inputs influence the generated content.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`base`**
    - The base text input for the grid's overall theme or background. This input sets the foundational conditioning layer upon which additional cell-specific conditionings are applied.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `Conditioning`
## Output types
- **`conditioning`**
    - Comfy dtype: `CONDITIONING`
    - The resulting conditioning for the grid, ready to be used for generating content within each cell based on the provided text inputs.
    - Python dtype: `Conditioning`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ConditioningGridCond:
    """
    Does the job of multiple area conditions of the same size adjacent to each other.
    Saves space, and is easier and quicker to set up and modify.


    Inputs related notes
    ----------
    base : conditioning
        for most cases, you can set the base from a ClipTextEncode with an empty string.
        If you wish to have something between the cells as common ground, lower the strength and set
        the base with the shared elements.
    columns and rows : integer
        after setting the desired grid size, call the menu option "update inputs" to update
        the node's conditioning input sockets.

        In most cases, columns and rows, should not be converted to input.

        dev note: I've considered disabling columns and rows options to convert to input
        on the javascript side, which (that I am aware) could be done with a modification
        to the core/WidgetInputs.js -> isConvertableWidget(...).
        However, upon reflection, I think there may be use cases in which the inputs are set for the
        maximum size but only a selected number of columns or rows given via input are used.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "columns": grid_len_INPUT,
            "rows": grid_len_INPUT,
            "width": ("INT", {"default": 256, "min": 16, "max": 2048, "step": 1}),
            "height": ("INT", {"default": 256, "min": 16, "max": 2048, "step": 1}),
            "strength": ("FLOAT", {"default": 3, }),
            "base": ("CONDITIONING",)
        }}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "set_conditioning"
    CATEGORY = conditioning_category_path

    def set_conditioning(self, base, columns, rows, width, height, strength, **kwargs):
        cond = base
        cond_set_area_node = nodes.ConditioningSetArea()
        cond_combine_node = nodes.ConditioningCombine()

        for r in range(rows):
            for c in range(columns):
                arg_name = f"r{r + 1}_c{c + 1}"
                new_cond = kwargs[arg_name]
                new_cond_area = cond_set_area_node.append(new_cond, width, height, c * width, r * height, strength)[0]
                new_cond = cond_combine_node.combine(new_cond_area, cond)[0]

                cond = new_cond
        return (cond,)

```
