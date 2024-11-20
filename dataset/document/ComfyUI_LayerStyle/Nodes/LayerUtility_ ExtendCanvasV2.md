# LayerUtility: ExtendCanvas V2
## Documentation
- Class name: `LayerUtility: ExtendCanvasV2`
- Category: `😺dzNodes/LayerUtility`
- Output node: `False`

The ExtendCanvasV2 node is designed to enhance the functionality of image manipulation by providing advanced capabilities for extending the canvas of an image. This node aims to offer more sophisticated options and flexibility compared to its predecessor, allowing for more precise and creative adjustments in image composition.
## Input types
### Required
- **`image`**
    - Specifies the original image to be manipulated. It is the primary input on which the canvas extension operations are performed.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`invert_mask`**
    - A boolean parameter that determines whether the mask applied to the image should be inverted, affecting how the canvas extension is applied in relation to the mask.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`top`**
    - Defines the number of pixels to extend the canvas at the top edge of the image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`bottom`**
    - Specifies the number of pixels to extend the canvas at the bottom edge of the image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`left`**
    - Indicates the number of pixels to extend the canvas on the left side of the image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`right`**
    - Determines the number of pixels to extend the canvas on the right side of the image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`color`**
    - Specifies the color to fill the extended canvas area, allowing for aesthetic customization of the added canvas space.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
- **`mask`**
    - An optional mask that can be applied to the image, influencing the areas of the canvas extension.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The resulting image after the canvas has been extended, reflecting the specified adjustments.
    - Python dtype: `torch.Tensor`
- **`mask`**
    - Comfy dtype: `MASK`
    - The mask applied to the extended canvas, if any, indicating areas of interest or exclusion in the extended space.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ExtendCanvasV2:

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
                "color": ("STRING", {"default": "#000000"}),
            },
            "optional": {
                "mask": ("MASK",),  #
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask")
    FUNCTION = 'extend_canvas_v2'
    CATEGORY = '😺dzNodes/LayerUtility'

    def extend_canvas_v2(self, image, invert_mask,
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
