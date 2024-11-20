# LayerColor: ColorAdapter
## Documentation
- Class name: `LayerColor: ColorAdapter`
- Category: `😺dzNodes/LayerColor`
- Output node: `False`

The ColorAdapter node is designed to adapt the color of an image to match a reference image's color scheme, applying a specified opacity level to blend the adapted image. It processes each image in a batch, adjusting colors to closely resemble the reference, and supports images with transparency by maintaining the alpha channel.
## Input types
### Required
- **`image`**
    - The input image or images to be color adapted. This parameter is crucial for defining the source images whose colors need to be adjusted to match the reference image.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`color_ref_image`**
    - The reference image or images used to adapt the color scheme of the input images. It plays a key role in determining the target color profile for the adaptation process.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`opacity`**
    - The opacity level applied to the blended image, affecting the intensity of the color adaptation. It allows for fine-tuning the strength of the color match between the input and reference images.
    - Comfy dtype: `INT`
    - Python dtype: `float`
### Optional
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The batch of images after color adaptation, blended with the original images at the specified opacity level. It represents the final output where each image's color has been adjusted to match the reference.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ColorAdapter:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "image": ("IMAGE", ),  #
                "color_ref_image": ("IMAGE", ),  #
                "opacity": ("INT", {"default": 75, "min": 0, "max": 100, "step": 1}),  # 透明度
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'color_adapter'
    CATEGORY = '😺dzNodes/LayerColor'

    def color_adapter(self, image, color_ref_image, opacity):
        ret_images = []

        l_images = []
        r_images = []
        for l in image:
            l_images.append(torch.unsqueeze(l, 0))
        for r in color_ref_image:
            r_images.append(torch.unsqueeze(r, 0))
        for i in range(len(l_images)):
            _image = l_images[i]
            _ref = r_images[i] if len(ret_images) > i else r_images[-1]

            __image = tensor2pil(_image)
            _canvas = __image.convert('RGB')
            ret_image = color_adapter(_canvas, tensor2pil(_ref).convert('RGB'))
            ret_image = chop_image(_canvas, ret_image, blend_mode='normal', opacity=opacity)
            if __image.mode == 'RGBA':
                ret_image = RGB2RGBA(ret_image, __image.split()[-1])
            ret_images.append(pil2tensor(ret_image))

        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0),)

```
