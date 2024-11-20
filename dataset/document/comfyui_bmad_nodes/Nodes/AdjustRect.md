---
tags:
- Contour
- Image
---

# AdjustRect
## Documentation
- Class name: `AdjustRect`
- Category: `Bmad`
- Output node: `False`

The AdjustRect node is designed to modify the dimensions of a rectangle based on specified parameters, ensuring the new rectangle adheres to certain constraints like center alignment and size adjustments. It abstracts the complexity of geometric transformations, offering a straightforward way to recalibrate rectangle dimensions for various applications.
## Input types
### Required
- **`a`**
    - Represents one of the coordinates (x or y) of the rectangle, contributing to defining its initial position.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`b`**
    - Represents another coordinate (x or y) of the rectangle, working alongside 'a' to specify the rectangle's starting dimensions.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`c`**
    - Defines one of the dimensions (width or height) of the rectangle, influencing its size before adjustment.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`d`**
    - Defines another dimension (width or height) of the rectangle, used in conjunction with 'c' to determine the initial size.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`xm`**
    - Specifies the factor by which the rectangle's width is adjusted, ensuring the new width is a multiple of this value.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`ym`**
    - Specifies the factor by which the rectangle's height is adjusted, ensuring the new height is a multiple of this value.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`round_mode`**
    - Determines the rounding method used when adjusting the rectangle's dimensions, affecting the precision of the adjustment.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`input_format`**
    - Indicates the format of the input rectangle coordinates, dictating how they should be interpreted and transformed.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`output_format`**
    - Specifies the format for the output rectangle coordinates, determining how the adjusted dimensions are represented.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`int`**
    - Comfy dtype: `INT`
    - Each element of the output tuple represents a coordinate or dimension of the adjusted rectangle, reflecting the modifications applied to the original dimensions. The output is a tuple of integers, each specifying a part of the rectangle's adjusted geometry.
    - Python dtype: `Tuple[int, int, int, int]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class AdjustRect:
    round_mode_map = {
        'Floor': math.floor,  # may be close to the image's edge, keep rect tight
        'Ceil': math.ceil,  # need the margin and image's edges aren't near
        'Round': round,  # whatever fits closest to the original rect
        'Exact': lambda v: 1  # force exact measurement
    }
    round_modes = list(round_mode_map.keys())

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "a": ("INT", {"min": 0, "max": np.iinfo(np.int32).max, "step": 1}),
            "b": ("INT", {"min": 0, "max": np.iinfo(np.int32).max, "step": 1}),
            "c": ("INT", {"min": 0, "max": np.iinfo(np.int32).max, "step": 1}),
            "d": ("INT", {"min": 0, "max": np.iinfo(np.int32).max, "step": 1}),
            "xm": ("INT", {"default": 64, "min": 2, "max": 1280, "step": 2}),
            "ym": ("INT", {"default": 64, "min": 2, "max": 1280, "step": 2}),
            "round_mode": (cls.round_modes, {"default": cls.round_modes[2]}),
            "input_format": (rect_modes, {"default": rect_modes[1]}),
            "output_format": (rect_modes, {"default": rect_modes[1]}),
        }}

    RETURN_TYPES = tuple(["INT" for x in range(4)])
    FUNCTION = "adjust"
    CATEGORY = base_category_path

    def adjust(self, a, b, c, d, xm, ym, round_mode, input_format, output_format):
        x1, y1, x2, y2 = rect_modes_map[input_format]["toBounds"](a, b, c, d)
        center_x = (x1 + x2) // 2 + 1
        center_y = (y1 + y2) // 2 + 1
        # reasoning:
        # 00 01 02 03 04 05
        # __ -- -- -- -- __ ( original len of 4 )
        # __ x1 __ cx __ x2 ( target len of 4   )
        # most crop implementations include x1 but exclude x2;
        # thus is closer to original input
        # note that xm and ym are always even

        half_new_len_x = self.round_mode_map[round_mode]((x2 - x1) / xm) * xm // 2
        half_new_len_y = self.round_mode_map[round_mode]((y2 - y1) / ym) * ym // 2

        # note: these points can fall outside the image space
        x2 = x1 = center_x
        x2 += half_new_len_x
        x1 -= half_new_len_x
        y2 = y1 = center_y
        y2 += half_new_len_y
        y1 -= half_new_len_y

        # convert to desired output format
        x1, y1, x2, y2 = rect_modes_map[output_format]["fromBounds"](x1, y1, x2, y2)

        return (x1, y1, x2, y2,)

```
