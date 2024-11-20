---
tags:
- ImageSize
- ImageTransformation
---

# LayerUtility: ImageScaleRestore
## Documentation
- Class name: `LayerUtility: ImageScaleRestore`
- Category: `😺dzNodes/LayerUtility`
- Output node: `False`

This node is designed to adjust the scale of images to a specified size or restore them to their original dimensions. It supports various resizing methods and can handle images with or without associated masks, making it versatile for different image processing tasks.
## Input types
### Required
- **`image`**
    - The image to be scaled or restored. It is central to the node's operation as it directly influences the output image's dimensions and quality.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`scale`**
    - Determines the scaling factor or the target size for the image resizing. It plays a crucial role in defining the new dimensions of the image.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float or tuple`
- **`method`**
    - Specifies the algorithm used for resizing the image, affecting the quality and characteristics of the output.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`scale_by_longest_side`**
    - A flag indicating whether to scale the image by its longest side, ensuring aspect ratio preservation.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`longest_side`**
    - Specifies the target length of the longest side of the image when scaled, relevant only if 'scale_by_longest_side' is True.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`mask`**
    - An optional mask that will be resized in tandem with the image, useful for maintaining consistency between the image and its mask.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`original_size`**
    - The original dimensions of the image, used when restoring the image to its initial size or to scale the image to a specific size.
    - Comfy dtype: `BOX`
    - Python dtype: `tuple`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The scaled or restored image, reflecting the specified dimensions and resizing method.
    - Python dtype: `torch.Tensor`
- **`mask`**
    - Comfy dtype: `MASK`
    - The resized mask corresponding to the output image, if an input mask was provided.
    - Python dtype: `torch.Tensor`
- **`original_size`**
    - Comfy dtype: `BOX`
    - The original dimensions of the image, useful for reference or further processing.
    - Python dtype: `tuple`
- **`width`**
    - Comfy dtype: `INT`
    - The width of the output image after scaling, providing specific detail on the image's dimensions.
    - Python dtype: `int`
- **`height`**
    - Comfy dtype: `INT`
    - The height of the output image after scaling, offering precise information on the image's size.
    - Python dtype: `int`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ImageScaleRestore:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        method_mode = ['lanczos', 'bicubic', 'hamming', 'bilinear', 'box', 'nearest']
        return {
            "required": {
                "image": ("IMAGE", ),  #
                "scale": ("FLOAT", {"default": 1, "min": 0.01, "max": 100, "step": 0.01}),
                "method": (method_mode,),
                "scale_by_longest_side": ("BOOLEAN", {"default": False}),  # 是否按长边缩放
                "longest_side": ("INT", {"default": 1024, "min": 4, "max": 999999, "step": 1}),
            },
            "optional": {
                "mask": ("MASK",),  #
                "original_size": ("BOX",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "BOX", "INT", "INT")
    RETURN_NAMES = ("image", "mask", "original_size", "width", "height",)
    FUNCTION = 'image_scale_restore'
    CATEGORY = '😺dzNodes/LayerUtility'

    def image_scale_restore(self, image, scale, method,
                            scale_by_longest_side, longest_side,
                            mask = None,  original_size = None
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
                l_masks.append(tensor2pil(torch.unsqueeze(m, 0)).convert('L'))

        max_batch = max(len(l_images), len(l_masks))

        orig_width, orig_height = tensor2pil(l_images[0]).size
        if original_size is not None:
            target_width = original_size[0]
            target_height = original_size[1]
        else:
            target_width = int(orig_width * scale)
            target_height = int(orig_height * scale)
            if scale_by_longest_side:
                if orig_width > orig_height:
                    target_width = longest_side
                    target_height = int(target_width * orig_height / orig_width)
                else:
                    target_height = longest_side
                    target_width = int(target_height * orig_width / orig_height)
        if target_width < 4:
            target_width = 4
        if target_height < 4:
            target_height = 4
        resize_sampler = Image.LANCZOS
        if method == "bicubic":
            resize_sampler = Image.BICUBIC
        elif method == "hamming":
            resize_sampler = Image.HAMMING
        elif method == "bilinear":
            resize_sampler = Image.BILINEAR
        elif method == "box":
            resize_sampler = Image.BOX
        elif method == "nearest":
            resize_sampler = Image.NEAREST

        for i in range(max_batch):

            _image = l_images[i] if i < len(l_images) else l_images[-1]

            _canvas = tensor2pil(_image).convert('RGB')
            ret_image = _canvas.resize((target_width, target_height), resize_sampler)
            ret_mask = Image.new('L', size=ret_image.size, color='white')
            if mask is not None:
                _mask = l_masks[i] if i < len(l_masks) else l_masks[-1]
                ret_mask = _mask.resize((target_width, target_height), resize_sampler)

            ret_images.append(pil2tensor(ret_image))
            ret_masks.append(image2mask(ret_mask))

        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0), [orig_width, orig_height], target_width, target_height,)

```
