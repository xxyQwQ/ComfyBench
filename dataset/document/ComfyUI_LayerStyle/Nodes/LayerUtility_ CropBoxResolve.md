# LayerUtility: CropBoxResolve
## Documentation
- Class name: `LayerUtility: CropBoxResolve`
- Category: `😺dzNodes/LayerUtility`
- Output node: `False`

The CropBoxResolve node is designed to calculate the dimensions of a new, adjusted bounding box based on the coordinates of an existing crop box. It transforms the original coordinates into a format suitable for further image processing tasks, such as cropping or resizing.
## Input types
### Required
- **`crop_box`**
    - The 'crop_box' parameter represents the original bounding box coordinates. It is essential for determining the starting point and size of the area to be processed, affecting the output dimensions directly.
    - Comfy dtype: `BOX`
    - Python dtype: `Tuple[int, int, int, int]`
### Optional
## Output types
- **`x`**
    - Comfy dtype: `INT`
    - The 'x' output represents the starting x-coordinate of the new bounding box.
    - Python dtype: `int`
- **`y`**
    - Comfy dtype: `INT`
    - The 'y' output represents the starting y-coordinate of the new bounding box.
    - Python dtype: `int`
- **`width`**
    - Comfy dtype: `INT`
    - The 'width' output specifies the width of the new bounding box.
    - Python dtype: `int`
- **`height`**
    - Comfy dtype: `INT`
    - The 'height' output specifies the height of the new bounding box.
    - Python dtype: `int`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class CropBoxResolve:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "crop_box": ("BOX",),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT")
    RETURN_NAMES = ("x", "y", "width", "height")
    FUNCTION = 'crop_box_resolve'
    CATEGORY = '😺dzNodes/LayerUtility'

    def crop_box_resolve(self, crop_box
                  ):

        (x1, y1, x2, y2) = crop_box
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        return (x1, y1, x2 - x1, y2 - y1,)

```
