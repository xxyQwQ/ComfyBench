---
tags:
- Mask
---

# LayerMask: RmBgUltra V2
## Documentation
- Class name: `LayerMask: RmBgUltra V2`
- Category: `😺dzNodes/LayerMask`
- Output node: `False`

This node specializes in removing backgrounds from images with an enhanced version of the algorithm, aiming for higher accuracy and detail preservation. It processes images to extract foreground elements by applying advanced techniques to improve edge definition and reduce artifacts, making it suitable for complex image compositions.
## Input types
### Required
- **`image`**
    - The input image or images to be processed for background removal. This parameter is crucial as it directly influences the effectiveness of the background removal process.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`detail_method`**
    - Specifies the method used to enhance or preserve detail in the image during the background removal process, affecting the precision of edge handling.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`detail_erode`**
    - Defines the amount of erosion applied to the detail in the image, influencing the edge softness and detail preservation.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`detail_dilate`**
    - Specifies the amount of dilation applied to the detail in the image, affecting the edge sharpness and detail enhancement.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`black_point`**
    - Defines the black point threshold for adjusting the contrast of the mask, influencing the visibility of darker areas in the resulting image.
    - Comfy dtype: `FLOAT`
    - Python dtype: `int`
- **`white_point`**
    - Sets the white point threshold for mask contrast adjustment, impacting the brightness and visibility of lighter areas in the output.
    - Comfy dtype: `FLOAT`
    - Python dtype: `int`
- **`process_detail`**
    - A boolean flag indicating whether to apply additional processing to enhance image details, affecting the overall quality and clarity of the foreground extraction.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`device`**
    - Specifies the device (CPU or GPU) on which the node's computations are performed, affecting performance and efficiency.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`max_megapixels`**
    - Limits the maximum size of images processed by the node to prevent excessive memory usage, affecting processing speed and resource allocation.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
### Optional
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The collection of images with the background removed, showcasing the foreground elements with enhanced clarity and detail.
    - Python dtype: `torch.Tensor`
- **`mask`**
    - Comfy dtype: `MASK`
    - The corresponding masks indicating the areas of the image where the background has been removed, useful for further image processing tasks.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class RmBgUltraV2:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):

        method_list = ['VITMatte', 'VITMatte(local)', 'PyMatting', 'GuidedFilter', ]
        device_list = ['cuda','cpu']
        return {
            "required": {
                "image": ("IMAGE",),
                "detail_method": (method_list,),
                "detail_erode": ("INT", {"default": 6, "min": 1, "max": 255, "step": 1}),
                "detail_dilate": ("INT", {"default": 6, "min": 1, "max": 255, "step": 1}),
                "black_point": ("FLOAT", {"default": 0.01, "min": 0.01, "max": 0.98, "step": 0.01, "display": "slider"}),
                "white_point": ("FLOAT", {"default": 0.99, "min": 0.02, "max": 0.99, "step": 0.01, "display": "slider"}),
                "process_detail": ("BOOLEAN", {"default": True}),
                "device": (device_list,),
                "max_megapixels": ("FLOAT", {"default": 2.0, "min": 1, "max": 999, "step": 0.1}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", )
    RETURN_NAMES = ("image", "mask", )
    FUNCTION = "rmbg_ultra_v2"
    CATEGORY = '😺dzNodes/LayerMask'
  
    def rmbg_ultra_v2(self, image, detail_method, detail_erode, detail_dilate,
                       black_point, white_point, process_detail, device, max_megapixels):
        ret_images = []
        ret_masks = []

        if detail_method == 'VITMatte(local)':
            local_files_only = True
        else:
            local_files_only = False

        for i in image:
            i = torch.unsqueeze(i, 0)
            i = pil2tensor(tensor2pil(i).convert('RGB'))
            orig_image = tensor2pil(i).convert('RGB')
            _mask = RMBG(orig_image)
            _mask = pil2tensor(_mask)

            detail_range = detail_erode + detail_dilate
            if process_detail:
                if detail_method == 'GuidedFilter':
                    _mask = guided_filter_alpha(i, _mask, detail_range // 6 + 1)
                    _mask = tensor2pil(histogram_remap(_mask, black_point, white_point))
                elif detail_method == 'PyMatting':
                    _mask = tensor2pil(mask_edge_detail(i, _mask, detail_range // 8 + 1, black_point, white_point))
                else:
                    _trimap = generate_VITMatte_trimap(_mask, detail_erode, detail_dilate)
                    _mask = generate_VITMatte(orig_image, _trimap, local_files_only=local_files_only, device=device, max_megapixels=max_megapixels)
                    _mask = tensor2pil(histogram_remap(pil2tensor(_mask), black_point, white_point))
            else:
                _mask = mask2image(_mask)

            ret_image = RGB2RGBA(orig_image, _mask.convert('L'))
            ret_images.append(pil2tensor(ret_image))
            ret_masks.append(image2mask(_mask))

        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0),)

```
