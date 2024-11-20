# LayerUtility: GetImageSize
## Documentation
- Class name: `LayerUtility: GetImageSize`
- Category: `😺dzNodes/LayerUtility`
- Output node: `False`

This node is designed to determine the dimensions of an image, returning its width, height, and original size in a structured format. It abstracts the complexity of image processing to provide straightforward metrics that can be used for further image manipulation or analysis.
## Input types
### Required
- **`image`**
    - The 'image' parameter is the input image tensor for which the size is to be determined. It plays a crucial role in the node's operation by being the source from which the dimensions are extracted.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
### Optional
## Output types
- **`width`**
    - Comfy dtype: `INT`
    - The width of the input image as determined by the node.
    - Python dtype: `int`
- **`height`**
    - Comfy dtype: `INT`
    - The height of the input image as determined by the node.
    - Python dtype: `int`
- **`original_size`**
    - Comfy dtype: `BOX`
    - A tuple containing the original width and height of the input image, providing a compact representation of its dimensions.
    - Python dtype: `Tuple[int, int]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class GetImageSize:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "image": ("IMAGE", ),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("INT", "INT", "BOX")
    RETURN_NAMES = ("width", "height",  "original_size")
    FUNCTION = 'get_image_size'
    CATEGORY = '😺dzNodes/LayerUtility'

    def get_image_size(self, image,):

        if image.shape[0] > 0:
            image = torch.unsqueeze(image[0], 0)
        _image = tensor2pil(image)

        return (_image.width, _image.height, [_image.width, _image.height],)

```
