# LayerUtility: ImageRemoveAlpha
## Documentation
- Class name: `LayerUtility: ImageRemoveAlpha`
- Category: `😺dzNodes/LayerUtility`
- Output node: `False`

The ImageRemoveAlpha node is designed to process images by removing the alpha channel. It can either fill the background with a specified color or convert images directly to RGB, depending on whether the background filling option is enabled. This functionality is essential for image processing tasks where transparency needs to be handled or eliminated, such as preparing images for environments that do not support alpha channels.
## Input types
### Required
- **`RGBA_image`**
    - The RGBA_image parameter represents the input images with an alpha channel that needs processing. It is crucial for specifying the images on which the alpha removal or background filling operations will be performed.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`fill_background`**
    - The fill_background parameter determines whether the background of the image should be filled with a specified color when the alpha channel is removed. This affects whether the image will simply have its alpha channel discarded or if it will have a new background color applied.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`background_color`**
    - The background_color parameter specifies the color to use when filling the background after removing the alpha channel. It is essential for customizing the appearance of the output images when the fill_background option is enabled.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
- **`mask`**
    - The mask parameter allows for the specification of a custom mask to define how the alpha channel should be removed or how the background should be filled. It provides additional flexibility in processing images with complex transparency or background requirements.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
## Output types
- **`RGB_image`**
    - Comfy dtype: `IMAGE`
    - The RGB_image output represents the processed images with the alpha channel removed. These images are either directly converted to RGB or have their backgrounds filled with a specified color, depending on the node's configuration.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ImageRemoveAlpha:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "RGBA_image": ("IMAGE", ),  #
                "fill_background": ("BOOLEAN", {"default": False}),
                "background_color": ("STRING", {"default": "#000000"}),
            },
            "optional": {
                "mask": ("MASK",),  #
            }
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("RGB_image", )
    FUNCTION = 'image_remove_alpha'
    CATEGORY = '😺dzNodes/LayerUtility'

    def image_remove_alpha(self, RGBA_image, fill_background, background_color, mask=None):

        ret_images = []

        for index, img in enumerate(RGBA_image):
            _image = tensor2pil(img)

            if fill_background:
                if mask is not None:
                    m = mask[index].unsqueeze(0) if index < len(mask) else mask[-1].unsqueeze(0)
                    alpha = tensor2pil(m).convert('L')
                elif _image.mode == "RGBA":
                    alpha = _image.split()[-1]
                else:
                    log(f"Error: {NODE_NAME} skipped, because the input image is not RGBA and mask is None.",
                        message_type='error')
                    return (RGBA_image,)
                ret_image = Image.new('RGB', size=_image.size, color=background_color)
                ret_image.paste(_image, mask=alpha)
                ret_images.append(pil2tensor(ret_image))

            else:
                ret_images.append(pil2tensor(tensor2pil(img).convert('RGB')))

        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0), )

```
