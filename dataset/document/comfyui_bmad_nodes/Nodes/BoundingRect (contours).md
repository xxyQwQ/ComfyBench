---
tags:
- Contour
- Image
---

# BoundingRect (contours)
## Documentation
- Class name: `BoundingRect (contours)`
- Category: `Bmad/CV/Contour`
- Output node: `False`

This node calculates the bounding rectangle of a contour. It is designed to provide a simple way to obtain the smallest rectangle that can fully enclose a contour. This functionality is crucial for various computer vision tasks where understanding the spatial extent of objects within an image is necessary.
## Input types
### Required
- **`contour`**
    - A single contour for which the bounding rectangle will be calculated. This input is crucial for determining the spatial boundaries of the object represented by the contour.
    - Comfy dtype: `CV_CONTOUR`
    - Python dtype: `ndarray`
- **`return_mode`**
    - Specifies the mode of calculation for the bounding rectangle, affecting the format of the returned values.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`int`**
    - Comfy dtype: `INT`
    - The coordinates of the top-left corner and the dimensions of the bounding rectangle (x, y, width, height). This output provides essential spatial information about the contour.
    - Python dtype: `Tuple[int, int, int, int]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ContourGetBoundingRect:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "contour": ("CV_CONTOUR",),
                "return_mode": (rect_modes, {"default": rect_modes[1]})
            },
        }

    RETURN_TYPES = tuple(["INT" for _ in range(4)])
    FUNCTION = "compute"
    CATEGORY = f"{cv_category_path}/Contour"

    def compute(self, contour, return_mode):
        if contour is None:
            print("Contour = None !")
            return (0, 0, 0, 0,)

        # convert opencv boundingRect format to bounds
        bounds = rect_modes_map[rect_modes[0]]["toBounds"](*cv.boundingRect(contour))

        # convert from bounds to desired output format on return
        return rect_modes_map[return_mode]["fromBounds"](*bounds)

```
