# Image size to string
## Documentation
- Class name: `Image size to string`
- Category: `Hakkun`
- Output node: `False`

The node converts the dimensions of an image into a string representation, indicating the image's width and height in a concise format. It abstracts the process of extracting image dimensions and formatting them into a human-readable string.
## Input types
### Required
- **`image`**
    - The input image whose dimensions are to be converted into a string. This parameter is crucial for determining the size of the image to be represented as a string.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
## Output types
- **`size`**
    - Comfy dtype: `STRING`
    - A string representing the dimensions of the input image, formatted as 'widthxheight'.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ImageSizeToString:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (IMAGE_TYPE,),
            }
        }

    RETURN_TYPES = (TEXT_TYPE,)
    RETURN_NAMES = ("size",)
    FUNCTION = "calculate"

    CATEGORY = "Hakkun"

    def calculate(self, image):
        image_size = image.size()
        img_width = int(image_size[2])
        img_height = int(image_size[1])

        size = str(img_width)+'x'+str(img_height)

        return size,

```
