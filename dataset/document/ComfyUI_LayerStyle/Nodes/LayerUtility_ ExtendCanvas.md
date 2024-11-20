# LayerUtility: ExtendCanvas
## Documentation
- Class name: `LayerUtility: ExtendCanvas`
- Category: `😺dzNodes/LayerUtility`
- Output node: `False`

This node is designed to extend the canvas of an image, allowing for adjustments in size and dimensions without altering the original content. It provides a way to add space around an image, which can be useful for layout, design, or preparation for further processing.
## Input types
### Required
- **`image`**
    - The primary image to be extended. This parameter is crucial as it defines the base upon which the canvas will be extended.
    - Comfy dtype: `IMAGE`
    - Python dtype: `Image`
- **`invert_mask`**
    - Determines whether the mask should be inverted during the process. This can affect how the mask is applied in relation to the canvas extension.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`top`**
    - Specifies the amount of space to add to the top of the canvas. It affects the vertical expansion of the image's canvas.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`bottom`**
    - Determines the amount of space to add to the bottom of the canvas, contributing to the vertical expansion.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`left`**
    - Indicates the amount of space to add to the left side of the canvas, affecting the horizontal expansion.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`right`**
    - Specifies the amount of space to add to the right side of the canvas, also affecting the horizontal expansion.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`color`**
    - Defines the color to fill the newly added space around the image, ensuring the extension blends seamlessly or contrasts as desired.
    - Comfy dtype: `COLOR`
    - Python dtype: `str`
### Optional
- **`mask`**
    - An optional mask that can be applied to the image during the canvas extension process. It allows for more precise control over how the extension affects specific parts of the image.
    - Comfy dtype: `MASK`
    - Python dtype: `Image`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The resulting image after the canvas has been extended. It showcases the original image with added space around it as specified by the input parameters.
    - Python dtype: `Image`
- **`mask`**
    - Comfy dtype: `MASK`
    - The mask corresponding to the extended canvas, which can be used for further image processing or manipulation.
    - Python dtype: `Image`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ExtendCanvas:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "image": ("IMAGE", ),
                "invert_mask": ("BOOLEAN", {"default": True}),  # 反转mask
                "top": ("INT", {"default": 0, "min": 0, "max": 99999, "step": 1}),
                "bottom": ("INT", {"default": 0, "min": 0, "max": 99999, "step": 1}),
                "left": ("INT", {"default": 0, "min": 0, "max": 99999, "step": 1}),
                "right": ("INT", {"default": 0, "min": 0, "max": 99999, "step": 1}),
                "color": ("COLOR", {"default": "#000000"},),
            },
            "optional": {
                "mask": ("MASK",),  #
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask")
    FUNCTION = 'extend_canvas'
    CATEGORY = '😺dzNodes/LayerUtility'

    def extend_canvas(self, image, invert_mask,
                  top, bottom, left, right, color,
                  mask=None,
                  ):

        l_images = []
        l_masks = []
        ret_images = []
        ret_masks = []

        for l in image:
            l_images.append(torch.unsqueeze(l, 0))
            m = tensor2pil(l)
            if m.mode == 'RGBA':
                l_masks.append(m.split()[-1])

        if mask is not None:
            if mask.dim() == 2:
                mask = torch.unsqueeze(mask, 0)
            l_masks = []
            for m in mask:
                if invert_mask:
                    m = 1 - m
                l_masks.append(tensor2pil(torch.unsqueeze(m, 0)).convert('L'))
        else:
            if len(l_masks) == 0:
                l_masks.append(Image.new('L', size=tensor2pil(l_images[0]).size, color='white'))

        max_batch = max(len(l_images), len(l_masks))
        for i in range(max_batch):

            _image = l_images[i] if i < len(l_images) else l_images[-1]
            _image = tensor2pil(_image).convert('RGB')
            _mask = l_masks[i] if i < len(l_masks) else l_masks[-1]

            width = _image.width + left + right
            height = _image.height + top + bottom
            _canvas = Image.new('RGB', (width, height), color)
            _mask_canvas = Image.new('L', (width, height), "black")

            _canvas.paste(_image, box=(left,top))
            _mask_canvas.paste(_mask.convert('L'), box=(left, top))

            ret_images.append(pil2tensor(_canvas))
            ret_masks.append(image2mask(_mask_canvas))

        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0),)

```
