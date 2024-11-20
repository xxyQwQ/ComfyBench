---
tags:
- Conditioning
---

# Conditioning Grid (string) Advanced
## Documentation
- Class name: `Conditioning Grid (string) Advanced`
- Category: `Bmad/conditioning`
- Output node: `False`

This node enhances the functionality of ConditioningGridCond by incorporating an advanced text encoding step for each text input using AdvancedCLIPTextEncode. It automates the process of encoding text inputs into conditionings and then applies these conditionings to a grid's AreaConditioners, facilitating a more nuanced and customizable conditioning process for grid-based operations.
## Input types
### Required
- **`clip`**
    - The CLIP model used for encoding the text inputs. It plays a crucial role in determining the quality and relevance of the generated conditionings.
    - Comfy dtype: `CLIP`
    - Python dtype: `comfy.sd.CLIP`
- **`base`**
    - The base text input that is encoded and used as the foundational conditioning for the grid. It sets the initial context or theme for the grid's content.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`columns`**
    - Specifies the number of columns in the grid. This determines the grid's horizontal dimension and how many conditionings are applied across it.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`rows`**
    - Specifies the number of rows in the grid. This determines the grid's vertical dimension and how many conditionings are applied along it.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`width`**
    - The width of each grid cell. This affects the spatial resolution of the conditionings applied within the grid.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - The height of each grid cell. This affects the spatial resolution of the conditionings applied within the grid.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`strength`**
    - Determines the intensity or influence of the conditioning on the grid. It modulates how strongly the encoded texts affect the grid's output.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`token_normalization`**
    - Specifies the method for normalizing the tokens in the text encoding process, affecting the conditioning's consistency and balance.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`weight_interpretation`**
    - Defines how the weights are interpreted in the text encoding process, influencing the emphasis and distribution of the conditioning across the grid.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`conditioning`**
    - Comfy dtype: `CONDITIONING`
    - The resulting conditioning that combines the encoded base text and grid-specific conditionings, ready for further processing or application.
    - Python dtype: `custom.Conditioning`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ConditioningGridStr_ADVEncode:
    """
    Node similar to ConditioningGridCond, but automates an additional step, using a ClipTextEncode per text input.
    Each conditioning obtained from the text inputs is then used as input for the Grid's AreaConditioners.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "clip": ("CLIP",),
            "base": ("STRING", {"default": '', "multiline": False}),
            "columns": grid_len_INPUT,
            "rows": grid_len_INPUT,
            "width": ("INT", {"default": 256, "min": 16, "max": 2048, "step": 1}),
            "height": ("INT", {"default": 256, "min": 16, "max": 2048, "step": 1}),
            "strength": ("FLOAT", {"default": 3, }),

            "token_normalization": (["none", "mean", "length", "length+mean"],),
            "weight_interpretation": (["comfy", "A1111", "compel", "comfy++", "down_weight"],)
        }}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "set_conditioning"
    CATEGORY = conditioning_category_path

    #def encode(self, clip: comfy.sd.CLIP, text: str, parser: str, mean_normalization: bool, multi_conditioning: bool, use_old_emphasis_implementation: bool, use_CFGDenoiser:bool,with_SDXL=False,text_g="",text_l=""):
    def set_conditioning(self, clip, base, columns, rows, width, height, strength,
                         token_normalization, weight_interpretation,
                         **kwargs):
        text_encode_node = AdvancedCLIPTextEncode()
        cond_grid_node = ConditioningGridCond()

        encoded_base = text_encode_node.encode(clip, base, token_normalization, weight_interpretation, 'disable')[0]
        encoded_grid = {}
        for r in range(rows):
            for c in range(columns):
                cell = f"r{r + 1}_c{c + 1}"
                encoded_grid[cell] = \
                text_encode_node.encode(clip, kwargs[cell], token_normalization, weight_interpretation, 'disable')[0]

        return cond_grid_node.set_conditioning(encoded_base, columns, rows, width, height, strength, **encoded_grid)

```
