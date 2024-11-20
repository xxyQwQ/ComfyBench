---
tags:
- Color
---

# LayerColor: Exposure
## Documentation
- Class name: `LayerColor: Exposure`
- Category: `😺dzNodes/LayerColor`
- Output node: `False`

The Exposure node is designed to adjust the exposure level of images, allowing for the brightening or darkening of the image based on the specified exposure value. This node applies a mathematical transformation to the pixel values to simulate the effect of changing the camera's exposure setting, enhancing or reducing the image's overall luminance.
## Input types
### Required
- **`image`**
    - The 'image' parameter represents the input image(s) to be adjusted. It is crucial for determining the base visual content that will undergo exposure correction, directly influencing the outcome of the node's processing.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`exposure`**
    - The 'exposure' parameter specifies the degree to which the image's exposure is adjusted. A positive value increases brightness, while a negative value decreases it, significantly affecting the visual result of the image processing.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output 'image' is the result of applying the exposure correction to the input image(s), reflecting the adjustments made based on the specified exposure level.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ColorCorrectExposure:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "image": ("IMAGE", ),  #
                "exposure": ("INT", {"default": 20, "min": -100, "max": 100, "step": 1}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'color_correct_exposure'
    CATEGORY = '😺dzNodes/LayerColor'

    def color_correct_exposure(self, image, exposure):

        ret_images = []

        for i in image:
            i = torch.unsqueeze(i, 0)
            __image = tensor2pil(i)
            t = i.detach().clone().cpu().numpy().astype(np.float32)
            more = t[:, :, :, :3] > 0
            t[:, :, :, :3][more] *= pow(2, exposure / 32)
            if exposure < 0:
                bp = -exposure / 250
                scale = 1 / (1 - bp)
                t = np.clip((t - bp) * scale, 0.0, 1.0)
            ret_image = tensor2pil(torch.from_numpy(t))

            if __image.mode == 'RGBA':
                ret_image = RGB2RGBA(ret_image, __image.split()[-1])

            ret_images.append(pil2tensor(ret_image))

        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0),)

```
