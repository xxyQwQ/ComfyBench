# LayerFilter: GaussianBlur
## Documentation
- Class name: `LayerFilter: GaussianBlur`
- Category: `😺dzNodes/LayerFilter`
- Output node: `False`

The GaussianBlur node applies a Gaussian blur filter to images, effectively smoothing them by reducing high-frequency noise and details. This node is part of the LayerFilter category, focusing on enhancing or modifying the visual appearance of images through blurring techniques.
## Input types
### Required
- **`image`**
    - The 'image' parameter represents the input image(s) to be blurred. It is crucial for defining the visual content that will undergo the Gaussian blur transformation.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`blur`**
    - The 'blur' parameter specifies the intensity of the Gaussian blur effect. A higher value results in a more pronounced blurring effect, directly influencing the smoothness of the output image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output 'image' parameter is the result of applying the Gaussian blur filter to the input image(s), showcasing the transformed visual content with reduced noise and details.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class GaussianBlur:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "image": ("IMAGE", ),  #
                "blur": ("INT", {"default": 20, "min": 1, "max": 999, "step": 1}),  # 模糊
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'gaussian_blur'
    CATEGORY = '😺dzNodes/LayerFilter'

    def gaussian_blur(self, image, blur):

        ret_images = []

        for i in image:
            _canvas = tensor2pil(torch.unsqueeze(i, 0)).convert('RGB')

            ret_images.append(pil2tensor(gaussian_blur(_canvas, blur)))

        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0),)

```
