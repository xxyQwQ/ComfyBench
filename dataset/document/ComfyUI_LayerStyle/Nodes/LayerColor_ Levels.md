---
tags:
- Color
---

# LayerColor: Levels
## Documentation
- Class name: `LayerColor: Levels`
- Category: `😺dzNodes/LayerColor`
- Output node: `False`

The 'LayerColor: Levels' node is designed for advanced color correction of images. It allows users to adjust the intensity levels of specific color channels, including black point, white point, and midtones (gray point), as well as to set the output range for these adjustments. This functionality is crucial for enhancing image contrast or correcting the color balance in images to achieve a desired aesthetic or to correct photographic errors.
## Input types
### Required
- **`image`**
    - The input image to be processed. This is the primary data on which color correction levels will be applied.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`channel`**
    - Specifies the color channel(s) (RGB, red, green, blue) to be adjusted. This allows for targeted color correction on specific channels, enhancing flexibility in image editing.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
- **`black_point`**
    - Sets the black point threshold for the selected channel(s). Adjusting this value can alter the darkest parts of the image, affecting overall contrast and detail in shadows.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`white_point`**
    - Defines the white point threshold for the selected channel(s). Modifying this value impacts the brightest parts of the image, influencing highlights and overall image brightness.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`gray_point`**
    - Adjusts the midtone (gray point) level for the selected channel(s). This is crucial for fine-tuning the balance between shadows and highlights, affecting the image's overall tonal range.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`output_black_point`**
    - Sets the output range's black point, allowing for further refinement of the image's contrast and depth.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`output_white_point`**
    - Determines the output range's white point, enabling adjustments to the image's brightness and contrast levels.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output image after applying the specified color correction levels. This image reflects the adjustments made to the intensity levels of the selected color channels.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ColorCorrectLevels:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        channel_list = ["RGB", "red", "green", "blue"]
        return {
            "required": {
                "image": ("IMAGE", ),  #
                "channel": (channel_list,),
                "black_point": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1, "display": "slider"}),
                "white_point": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1, "display": "slider"}),
                "gray_point": ("FLOAT", {"default": 1, "min": 0.01, "max": 9.99, "step": 0.01}),
                "output_black_point": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                "output_white_point": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'levels'
    CATEGORY = '😺dzNodes/LayerColor'

    def levels(self, image, channel,
                      black_point, white_point,
                      gray_point, output_black_point, output_white_point):

        l_images = []
        l_masks = []
        ret_images = []

        for l in image:
            l_images.append(torch.unsqueeze(l, 0))
            m = tensor2pil(l)
            if m.mode == 'RGBA':
                l_masks.append(m.split()[-1])
            else:
                l_masks.append(Image.new('L', m.size, 'white'))


        for i in range(len(l_images)):
            _image = l_images[i]
            _mask = l_masks[i]
            orig_image = tensor2pil(_image)


            if channel == "red":
                r, g, b, _ = image_channel_split(orig_image, 'RGB')
                r = adjust_levels(r, black_point, white_point, gray_point,
                                          output_black_point, output_white_point)
                ret_image = image_channel_merge((r.convert('L'), g, b), 'RGB')
            elif channel == "green":
                r, g, b, _ = image_channel_split(orig_image, 'RGB')
                g = adjust_levels(g, black_point, white_point, gray_point,
                                  output_black_point, output_white_point)
                ret_image = image_channel_merge((r, g.convert('L'), b), 'RGB')
            elif channel == "blue":
                r, g, b, _ = image_channel_split(orig_image, 'RGB')
                b = adjust_levels(b, black_point, white_point, gray_point,
                                  output_black_point, output_white_point)
                ret_image = image_channel_merge((r, g, b.convert('L')), 'RGB')
            else:
                ret_image = adjust_levels(orig_image, black_point, white_point, gray_point,
                                          output_black_point, output_white_point)

            if orig_image.mode == 'RGBA':
                ret_image = RGB2RGBA(ret_image, orig_image.split()[-1])

            ret_images.append(pil2tensor(ret_image))

        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0),)

```
