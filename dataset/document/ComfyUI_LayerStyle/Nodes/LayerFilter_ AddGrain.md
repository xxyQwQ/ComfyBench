# LayerFilter: Add Grain
## Documentation
- Class name: `LayerFilter: AddGrain`
- Category: `😺dzNodes/LayerFilter`
- Output node: `False`

The AddGrain node is designed to apply a grain effect to images, enhancing their texture by adding a layer of visual noise. This process aims to simulate the grainy appearance often found in film photography, thereby adding a stylistic or aesthetic quality to digital images.
## Input types
### Required
- **`image`**
    - The input image(s) to which the grain effect will be applied. This parameter is crucial for defining the base upon which the grain effect will be overlaid, directly influencing the node's output. The choice of images determines the initial visual context for the grain effect, impacting both the aesthetic and technical aspects of the final output.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`grain_power`**
    - Determines the intensity of the grain effect applied to the image, with higher values resulting in a more pronounced grain texture. This parameter allows for fine-tuning the visual impact of the grain on the image, significantly affecting the texture's visibility and overall feel of the processed image.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`grain_scale`**
    - Controls the scale of the grain particles, affecting their size and distribution across the image. This parameter enables customization of the grain effect to achieve desired visual outcomes, influencing how natural or pronounced the grain appears in relation to the image's content.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`grain_sat`**
    - Adjusts the saturation of the grain effect, influencing the color intensity of the grain particles. This parameter allows for the modulation of the grain's color characteristics to match the aesthetic needs of the image, affecting the colorfulness and vibrancy of the grain effect.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
### Optional
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output image(s) with the grain effect applied. This parameter represents the transformation of the input image(s) through the addition of grain, enhancing their texture and visual appeal.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class AddGrain:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "image": ("IMAGE", ),  #
                "grain_power": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01}),
                "grain_scale": ("FLOAT", {"default": 1, "min": 0.1, "max": 10, "step": 0.1}),
                "grain_sat": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'add_grain'
    CATEGORY = '😺dzNodes/LayerFilter'

    def add_grain(self, image, grain_power, grain_scale, grain_sat):

        ret_images = []

        for i in image:
            _canvas = tensor2pil(torch.unsqueeze(i, 0)).convert('RGB')
            _canvas = image_add_grain(_canvas, grain_scale, grain_power, grain_sat, toe=0, seed=int(time.time()))
            ret_images.append(pil2tensor(_canvas))

        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0),)

```
