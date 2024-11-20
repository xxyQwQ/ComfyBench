---
tags:
- Color
---

# FindComplementaryColor
## Documentation
- Class name: `FindComplementaryColor`
- Category: `Bmad/CV/Color A.`
- Output node: `False`

This node is designed to find a complementary color within an image based on a given color dictionary and a specified power level. It can optionally use a mask to limit the search area within the image. The node aims to enhance or modify images by identifying and utilizing complementary colors, thereby supporting tasks related to color theory and image editing.
## Input types
### Required
- **`image`**
    - The input image on which the complementary color search is performed. It serves as the primary data for analysis.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`color_dict`**
    - A dictionary mapping specific colors to their identifiers, used to determine the complementary color within the image.
    - Comfy dtype: `COLOR_DICT`
    - Python dtype: `Dict[str, Tuple[int, int, int]]`
- **`power`**
    - Affects the intensity of the search for a complementary color, influencing how the complementary color is determined.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
### Optional
- **`mask`**
    - An optional mask that can be applied to the image to restrict the search area for the complementary color.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
## Output types
- **`color`**
    - Comfy dtype: `COLOR`
    - The RGB values of the identified complementary color.
    - Python dtype: `List[int]`
- **`string`**
    - Comfy dtype: `STRING`
    - The identifier of the found complementary color from the provided color dictionary.
    - Python dtype: `str`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class FindComplementaryColor:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "color_dict": ("COLOR_DICT",),
            "power": ("FLOAT", {"default": 0.5, "min": .01, "max": 10, "step": "0.01"}),
        },
            "optional":
                {
                    "mask": ("IMAGE",)
                }
        }

    RETURN_TYPES = ("COLOR", "STRING",)
    FUNCTION = "find_color"
    CATEGORY = f"{cv_category_path}/Color A."

    def find_color(self, image, color_dict, power, mask=None):
        image = tensor2opencv(image, 3)

        if mask is not None:
            mask = tensor2opencv(mask, 1)

            # this is a quality of life feature, so that it is easier to run the node and test stuff
            # the behavior (img resize w/ lin. interpolation) can be avoided by setting up the data prior to this node
            if image.shape[0:2] != mask.shape[0:2]:
                print("FindComplementaryColor node will resize image to fit mask.")
                image = cv.resize(image, (mask.shape[1], mask.shape[0]), interpolation=cv.INTER_LINEAR)

        color = find_complementary_color(image, color_dict, mask, power)
        return (list(color_dict[color]), color,)

```
