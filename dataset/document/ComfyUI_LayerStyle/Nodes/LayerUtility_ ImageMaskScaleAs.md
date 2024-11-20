# LayerUtility: ImageMaskScaleAs
## Documentation
- Class name: `LayerUtility: ImageMaskScaleAs`
- Category: `😺dzNodes/LayerUtility`
- Output node: `False`

This node is designed to scale images and masks to match the dimensions of a target image or mask, applying various resizing methods and fitting options to ensure the output aligns with the specified dimensions. It supports different interpolation methods for resizing, offering flexibility in how the scaling is performed to meet the needs of different use cases.
## Input types
### Required
- **`scale_as`**
    - Specifies the target tensor to scale the image or mask as, determining the dimensions to which the inputs should be resized.
    - Comfy dtype: `*`
    - Python dtype: `torch.Tensor`
- **`fit`**
    - Defines how the image or mask should be fitted to the target dimensions, affecting the scaling and cropping behavior.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`method`**
    - Determines the interpolation method used for resizing, such as 'bicubic', 'nearest', etc., influencing the quality and characteristics of the output.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
### Optional
- **`image`**
    - The image tensor to be scaled. It is optional and when provided, it will be resized to match the target dimensions.
    - Comfy dtype: `IMAGE`
    - Python dtype: `Optional[torch.Tensor]`
- **`mask`**
    - The mask tensor to be scaled. It is optional and when provided, it will be resized alongside the image to match the target dimensions.
    - Comfy dtype: `MASK`
    - Python dtype: `Optional[torch.Tensor]`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The scaled image tensor, resized to match the target dimensions specified by 'scale_as'.
    - Python dtype: `Optional[torch.Tensor]`
- **`mask`**
    - Comfy dtype: `MASK`
    - The scaled mask tensor, resized alongside the image to match the target dimensions specified by 'scale_as'.
    - Python dtype: `Optional[torch.Tensor]`
- **`original_size`**
    - Comfy dtype: `BOX`
    - A list containing the original width and height of the input image or mask before scaling.
    - Python dtype: `List[int]`
- **`widht`**
    - Comfy dtype: `INT`
    - unknown
    - Python dtype: `unknown`
- **`height`**
    - Comfy dtype: `INT`
    - The target height to which the image or mask has been scaled.
    - Python dtype: `int`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ImageMaskScaleAs:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        fit_mode = ['letterbox', 'crop', 'fill']
        method_mode = ['lanczos', 'bicubic', 'hamming', 'bilinear', 'box', 'nearest']

        return {
            "required": {
                "scale_as": (any, {}),
                "fit": (fit_mode,),
                "method": (method_mode,),
            },
            "optional": {
                "image": ("IMAGE",),  #
                "mask": ("MASK",),  #
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "BOX", "INT", "INT")
    RETURN_NAMES = ("image", "mask", "original_size", "widht", "height",)
    FUNCTION = 'image_mask_scale_as'
    CATEGORY = '😺dzNodes/LayerUtility'

    def image_mask_scale_as(self, scale_as, fit, method,
                            image=None, mask = None,
                            ):
        if scale_as.shape[0] > 0:
            _asimage = tensor2pil(scale_as[0])
        else:
            _asimage = tensor2pil(scale_as)
        target_width, target_height = _asimage.size
        _mask = Image.new('L', size=_asimage.size, color='black')
        _image = Image.new('RGB', size=_asimage.size, color='black')
        orig_width = 4
        orig_height = 4
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

        ret_images = []
        ret_masks = []
        
        if image is not None:
            for i in image:
                i = torch.unsqueeze(i, 0)
                _image = tensor2pil(i).convert('RGB')
                orig_width, orig_height = _image.size
                _image = fit_resize_image(_image, target_width, target_height, fit, resize_sampler)
                ret_images.append(pil2tensor(_image))
        if mask is not None:
            if mask.dim() == 2:
                mask = torch.unsqueeze(mask, 0)
            for m in mask:
                m = torch.unsqueeze(m, 0)
                _mask = tensor2pil(m).convert('L')
                orig_width, orig_height = _mask.size
                _mask = fit_resize_image(_mask, target_width, target_height, fit, resize_sampler).convert('L')
                ret_masks.append(image2mask(_mask))
        if len(ret_images) > 0 and len(ret_masks) >0:
            log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
            return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0), [orig_width, orig_height],target_width, target_height,)
        elif len(ret_images) > 0 and len(ret_masks) == 0:
            log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
            return (torch.cat(ret_images, dim=0), None, [orig_width, orig_height],target_width, target_height,)
        elif len(ret_images) == 0 and len(ret_masks) > 0:
            log(f"{NODE_NAME} Processed {len(ret_masks)} image(s).", message_type='finish')
            return (None, torch.cat(ret_masks, dim=0), [orig_width, orig_height], target_width, target_height,)
        else:
            log(f"Error: {NODE_NAME} skipped, because the available image or mask is not found.", message_type='error')
            return (None, None, [orig_width, orig_height], 0, 0,)

```
