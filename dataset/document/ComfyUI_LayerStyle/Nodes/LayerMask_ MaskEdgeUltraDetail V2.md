---
tags:
- Mask
---

# LayerMask: MaskEdgeUltraDetail V2
## Documentation
- Class name: `LayerMask: MaskEdgeUltraDetail V2`
- Category: `😺dzNodes/LayerMask`
- Output node: `False`

This node specializes in enhancing the detail and precision of mask edges in layer masks, utilizing advanced techniques to refine edges for improved visual clarity and accuracy. It offers a sophisticated approach to mask edge processing, catering to needs for high-detail and precision in image editing tasks.
## Input types
### Required
- **`image`**
    - The input image(s) to be processed. This parameter is crucial for determining the areas where mask edge refinement is applied, directly influencing the outcome of the node's operation.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`mask`**
    - The mask(s) corresponding to the input image(s), indicating areas of interest for edge refinement. The precision of mask edges is enhanced based on this parameter, making it essential for targeted detail improvement.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`method`**
    - Specifies the technique used for mask edge refinement, affecting the level of detail and precision achieved in the processed mask edges.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`mask_grow`**
    - Determines the extent to which mask edges are expanded, playing a key role in the overall edge refinement process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`fix_gap`**
    - Controls the gap filling in mask edges, aiding in creating smoother and more continuous edges.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`fix_threshold`**
    - Sets the threshold for fixing gaps in mask edges, contributing to the refinement of edge detail and continuity.
    - Comfy dtype: `FLOAT`
    - Python dtype: `int`
- **`edge_erode`**
    - Defines the amount of erosion applied to mask edges, crucial for achieving the desired level of edge detail and precision.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`edte_dilate`**
    - Specifies the dilation applied to mask edges after erosion, balancing the edge refinement process by restoring some of the eroded details.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`black_point`**
    - Adjusts the black point in the image, affecting the contrast and visibility of mask edges.
    - Comfy dtype: `FLOAT`
    - Python dtype: `int`
- **`white_point`**
    - Adjusts the white point in the image, influencing the brightness and clarity of mask edges.
    - Comfy dtype: `FLOAT`
    - Python dtype: `int`
- **`device`**
    - Indicates the computing device (CPU or GPU) used for processing, affecting the performance and efficiency of the node.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`max_megapixels`**
    - Limits the maximum size of images processed, ensuring efficient memory usage and performance.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
### Optional
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The processed image(s) with refined mask edges, showcasing enhanced detail and precision.
    - Python dtype: `torch.Tensor`
- **`mask`**
    - Comfy dtype: `MASK`
    - The refined mask(s) with improved edge detail, corresponding to the processed image(s).
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class MaskEdgeUltraDetailV2:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):

        method_list = ['VITMatte', 'VITMatte(local)', 'PyMatting', 'GuidedFilter', ]
        device_list = ['cuda','cpu']
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "method": (method_list,),
                "mask_grow": ("INT", {"default": 0, "min": 0, "max": 256, "step": 1}),
                "fix_gap": ("INT", {"default": 0, "min": 0, "max": 32, "step": 1}),
                "fix_threshold": ("FLOAT", {"default": 0.75, "min": 0.01, "max": 0.99, "step": 0.01}),
                "edge_erode": ("INT", {"default": 6, "min": 1, "max": 255, "step": 1}),
                "edte_dilate": ("INT", {"default": 6, "min": 1, "max": 255, "step": 1}),
                "black_point": ("FLOAT", {"default": 0.01, "min": 0.01, "max": 0.98, "step": 0.01, "display": "slider"}),
                "white_point": ("FLOAT", {"default": 0.99, "min": 0.02, "max": 0.99, "step": 0.01, "display": "slider"}),
                "device": (device_list,),
                "max_megapixels": ("FLOAT", {"default": 2.0, "min": 1, "max": 999, "step": 0.1}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", )
    RETURN_NAMES = ("image", "mask", )
    FUNCTION = "mask_edge_ultra_detail_v2"
    CATEGORY = '😺dzNodes/LayerMask'
  
    def mask_edge_ultra_detail_v2(self, image, mask, method, mask_grow, fix_gap, fix_threshold,
                               edge_erode, edte_dilate, black_point, white_point, device, max_megapixels,):
        ret_images = []
        ret_masks = []
        l_images = []
        l_masks = []

        if method == 'VITMatte(local)':
            local_files_only = True
        else:
            local_files_only = False

        if mask.dim() == 2:
            mask = torch.unsqueeze(mask, 0)
        for l in image:
            l_images.append(torch.unsqueeze(l, 0))
        for m in mask:
            l_masks.append(torch.unsqueeze(m, 0))
        if len(l_images) != len(l_masks) or tensor2pil(l_images[0]).size != tensor2pil(l_masks[0]).size:
            log(f"Error: {NODE_NAME} skipped, because mask does'nt match image.", message_type='error')
            return (image, mask,)
        detail_range = edge_erode + edte_dilate
        for i in range(len(l_images)):
            _image = l_images[i]
            orig_image = tensor2pil(_image).convert('RGB')
            _image = pil2tensor(orig_image)
            _mask = l_masks[i]
            if mask_grow != 0:
                _mask = expand_mask(_mask, mask_grow, mask_grow//2)
            if fix_gap:
                _mask = mask_fix(_mask, 1, fix_gap, fix_threshold, fix_threshold)
            log(f"{NODE_NAME} Processing...")
            if method == 'GuidedFilter':
                _mask = guided_filter_alpha(_image, _mask, detail_range//6)
                _mask = tensor2pil(histogram_remap(_mask, black_point, white_point))
            elif method == 'PyMatting':
                _mask = tensor2pil(mask_edge_detail(_image, _mask, detail_range//8, black_point, white_point))
            else:
                _trimap = generate_VITMatte_trimap(_mask, edge_erode, edte_dilate)
                _mask = generate_VITMatte(orig_image, _trimap, local_files_only=local_files_only, device=device, max_megapixels=max_megapixels)
                _mask = tensor2pil(histogram_remap(pil2tensor(_mask), black_point, white_point))

            ret_image = RGB2RGBA(orig_image, _mask.convert('L'))
            ret_images.append(pil2tensor(ret_image))
            ret_masks.append(image2mask(_mask))

        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0),)

```
