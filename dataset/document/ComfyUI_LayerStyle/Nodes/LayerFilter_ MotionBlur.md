# LayerFilter: MotionBlur
## Documentation
- Class name: `LayerFilter: MotionBlur`
- Category: `😺dzNodes/LayerFilter`
- Output node: `False`

The MotionBlur node applies a directional blur effect to images, simulating the appearance of movement or speed. It allows for the adjustment of the blur's angle and intensity, enabling the creation of dynamic visual effects that convey motion.
## Input types
### Required
- **`image`**
    - The input image to which the motion blur effect will be applied. This parameter is crucial for defining the visual content that will undergo transformation.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`angle`**
    - Specifies the angle of the motion blur, allowing for the simulation of movement in various directions. This parameter is key to customizing the directionality of the perceived motion.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`blur`**
    - Determines the intensity of the motion blur effect, enabling control over how pronounced the motion appears in the image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output image with the motion blur effect applied, showcasing the simulated movement.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class MotionBlur:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "image": ("IMAGE", ),  #
                "angle": ("INT", {"default": 0, "min": -90, "max": 90, "step": 1}),  # 角度
                "blur": ("INT", {"default": 20, "min": 1, "max": 999, "step": 1}),  # 模糊
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'motion_blur'
    CATEGORY = '😺dzNodes/LayerFilter'

    def motion_blur(self, image, angle, blur):

        ret_images = []

        for i in image:

            _canvas = tensor2pil(torch.unsqueeze(i, 0)).convert('RGB')

            ret_images.append(pil2tensor(motion_blur(_canvas, angle, blur)))

        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0),)

```
