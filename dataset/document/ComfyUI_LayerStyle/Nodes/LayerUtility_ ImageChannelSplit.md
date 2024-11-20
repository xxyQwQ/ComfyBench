# LayerUtility: ImageChannelSplit
## Documentation
- Class name: `LayerUtility: ImageChannelSplit`
- Category: `😺dzNodes/LayerUtility`
- Output node: `False`

The ImageChannelSplit node is designed to split an input image into its constituent channels based on a specified color mode. It facilitates the manipulation of individual color channels for advanced image processing tasks, enabling users to work with each channel separately for detailed analysis or modification.
## Input types
### Required
- **`image`**
    - The input image to be split into its constituent channels. This parameter is crucial for the operation as it determines the source image for channel separation.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`mode`**
    - Specifies the color mode for channel splitting. This parameter affects how the input image is processed and determines the output channel format.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
### Optional
## Output types
- **`channel_1`**
    - Comfy dtype: `IMAGE`
    - The first channel of the split image, which varies depending on the specified mode.
    - Python dtype: `torch.Tensor`
- **`channel_2`**
    - Comfy dtype: `IMAGE`
    - The second channel of the split image, which varies depending on the specified mode.
    - Python dtype: `torch.Tensor`
- **`channel_3`**
    - Comfy dtype: `IMAGE`
    - The third channel of the split image, which varies depending on the specified mode.
    - Python dtype: `torch.Tensor`
- **`channel_4`**
    - Comfy dtype: `IMAGE`
    - The fourth channel of the split image, applicable in modes that support four channels.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ImageChannelSplit:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        channel_mode = ['RGBA', 'YCbCr', 'LAB', 'HSV']
        return {
            "required": {
                "image": ("IMAGE", ),  #
                "mode": (channel_mode,),  # 通道设置
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE",)
    RETURN_NAMES = ("channel_1", "channel_2", "channel_3", "channel_4",)
    FUNCTION = 'image_channel_split'
    CATEGORY = '😺dzNodes/LayerUtility'

    def image_channel_split(self, image, mode):

        c1_images = []
        c2_images = []
        c3_images = []
        c4_images = []

        for i in image:
            i = torch.unsqueeze(i, 0)
            _image = tensor2pil(i).convert('RGBA')
            channel1, channel2, channel3, channel4 = image_channel_split(_image, mode)
            c1_images.append(pil2tensor(channel1))
            c2_images.append(pil2tensor(channel2))
            c3_images.append(pil2tensor(channel3))
            c4_images.append(pil2tensor(channel4))

        log(f"{NODE_NAME} Processed {len(c1_images)} image(s).", message_type='finish')
        return (torch.cat(c1_images, dim=0), torch.cat(c2_images, dim=0), torch.cat(c3_images, dim=0), torch.cat(c4_images, dim=0),)

```
