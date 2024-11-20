# LayerFilter: ChannelShake
## Documentation
- Class name: `LayerFilter: ChannelShake`
- Category: `😺dzNodes/LayerFilter`
- Output node: `False`

The ChannelShake node applies a unique visual effect to images by shifting the color channels (RGB) in specified directions and distances. This manipulation creates a shaking or displacement effect on the image, offering a creative way to alter the visual appearance based on angle and distance parameters.
## Input types
### Required
- **`image`**
    - The input image to apply the channel shake effect. This parameter is crucial for defining the base image on which the effect will be applied.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`distance`**
    - Specifies the distance of the channel shift. This parameter influences the intensity of the shaking effect, with larger values resulting in more pronounced shifts.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`angle`**
    - Determines the angle at which the channels will be shifted. This affects the direction of the shake effect, allowing for a wide range of visual distortions.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`mode`**
    - Defines the order in which the RGB channels are shifted. This selection alters the visual outcome of the effect, offering various combinations for creative experimentation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
### Optional
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output image after applying the channel shake effect. This image showcases the visual transformation achieved through the specified channel shifts.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ChannelShake:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        channel_mode = ['RGB', 'RBG', 'BGR', 'BRG', 'GBR', 'GRB']
        return {
            "required": {
                "image": ("IMAGE", ),  #
                "distance": ("INT", {"default": 20, "min": 1, "max": 999, "step": 1}),  # 距离
                "angle": ("FLOAT", {"default": 40, "min": -360, "max": 360, "step": 0.1}),  # 角度
                "mode": (channel_mode,),  # 模式
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'channel_shake'
    CATEGORY = '😺dzNodes/LayerFilter'

    def channel_shake(self, image, distance, angle, mode, ):

        ret_images = []

        for i in image:
            i = torch.unsqueeze(i, 0)
            _canvas = tensor2pil(i).convert('RGB')
            R, G, B = _canvas.split()
            x = int(math.cos(angle) * distance)
            y = int(math.sin(angle) * distance)
            if mode.startswith('R'):
                R = shift_image(R.convert('RGB'), -x, -y).convert('L')
            if mode.startswith('G'):
                G = shift_image(G.convert('RGB'), -x, -y).convert('L')
            if mode.startswith('B'):
                B = shift_image(B.convert('RGB'), -x, -y).convert('L')
            if mode.endswith('R'):
                R = shift_image(R.convert('RGB'), x, y).convert('L')
            if mode.endswith('G'):
                G = shift_image(G.convert('RGB'), x, y).convert('L')
            if mode.endswith('B'):
                B = shift_image(B.convert('RGB'), x, y).convert('L')

            ret_image = Image.merge('RGB', [R, G, B])
            ret_images.append(pil2tensor(ret_image))

        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0),)

```
