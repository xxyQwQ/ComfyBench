---
tags:
- Mask
---

# LayerMask: MaskEdgeUltraDetail
## Documentation
- Class name: `LayerMask: MaskEdgeUltraDetail`
- Category: `😺dzNodes/LayerMask`
- Output node: `False`

This node specializes in enhancing the detail and precision of edges in layer masks. It applies advanced image processing techniques to refine mask edges, making them ultra-detailed and precise for improved visual quality in layered compositions.
## Input types
### Required
- **`image`**
    - The input image to be processed. This image serves as the base for mask edge enhancement, determining the areas where detail and precision are to be improved.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`mask`**
    - The mask that outlines the areas of the image to be detailed. It acts as a guide for the node to apply edge enhancement, defining the regions where ultra-detailing is necessary.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`method`**
    - Specifies the technique used for mask edge detailing. This parameter influences the approach taken to enhance the mask edges, affecting the outcome's precision and detail.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`mask_grow`**
    - Controls the expansion of the mask edges. This parameter adjusts how much the mask edges are grown to achieve the desired level of detail.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`fix_gap`**
    - Determines whether to fill gaps in the mask edges. This option helps in creating a more continuous and precise edge by bridging small gaps.
    - Comfy dtype: `INT`
    - Python dtype: `bool`
- **`fix_threshold`**
    - Sets the threshold for fixing gaps in the mask edges. It defines the sensitivity of gap detection and filling, impacting the smoothness of the final edge.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`detail_range`**
    - Defines the range of edge detailing. This parameter sets the extent to which the mask edges are enhanced, affecting the level of detail achieved.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`black_point`**
    - Adjusts the black point for edge detailing. This setting modifies the darkest areas of the mask, influencing the contrast and depth of the edge detail.
    - Comfy dtype: `FLOAT`
    - Python dtype: `int`
- **`white_point`**
    - Adjusts the white point for edge detailing. This setting modifies the lightest areas of the mask, influencing the brightness and highlight of the edge detail.
    - Comfy dtype: `FLOAT`
    - Python dtype: `int`
### Optional
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The processed image with enhanced mask edges. This output reflects the improvements in edge detail and precision, showcasing the node's effect on the original image.
    - Python dtype: `torch.Tensor`
- **`mask`**
    - Comfy dtype: `MASK`
    - The enhanced mask with ultra-detailed edges. This output demonstrates the refined precision and detail achieved through the node's processing.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class MaskEdgeUltraDetail:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        method_list = ['PyMatting', 'OpenCV-GuidedFilter']
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "method": (method_list,),
                "mask_grow": ("INT", {"default": 0, "min": -999, "max": 999, "step": 1}),
                "fix_gap": ("INT", {"default": 0, "min": 0, "max": 32, "step": 1}),
                "fix_threshold": ("FLOAT", {"default": 0.75, "min": 0.01, "max": 0.99, "step": 0.01}),
                "detail_range": ("INT", {"default": 12, "min": 1, "max": 256, "step": 1}),
                "black_point": ("FLOAT", {"default": 0.01, "min": 0.01, "max": 0.98, "step": 0.01}),
                "white_point": ("FLOAT", {"default": 0.99, "min": 0.02, "max": 0.99, "step": 0.01}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", )
    RETURN_NAMES = ("image", "mask", )
    FUNCTION = "mask_edge_ultra_detail"
    CATEGORY = '😺dzNodes/LayerMask'
  
    def mask_edge_ultra_detail(self, image, mask, method, mask_grow, fix_gap, fix_threshold,
                               detail_range, black_point, white_point,):
        ret_images = []
        ret_masks = []
        l_images = []
        l_masks = []
        if mask.dim() == 2:
            mask = torch.unsqueeze(mask, 0)
        for l in image:
            l_images.append(torch.unsqueeze(l, 0))
        for m in mask:
            l_masks.append(torch.unsqueeze(m, 0))
        if len(l_images) != len(l_masks) or tensor2pil(l_images[0]).size != tensor2pil(l_masks[0]).size:
            log(f"Error: {NODE_NAME} skipped, because mask does'nt match image.", message_type='error')
            return (image, mask,)

        for i in range(len(l_images)):
            _image = l_images[i]
            orig_image = tensor2pil(_image).convert('RGB')
            _image = pil2tensor(orig_image)
            _mask = l_masks[i]
            if mask_grow != 0:
                _mask = expand_mask(_mask, mask_grow, mask_grow//2)
            if fix_gap:
                _mask = mask_fix(_mask, 1, fix_gap, fix_threshold, fix_threshold)
            if method == 'OpenCV-GuidedFilter':
                _mask = guided_filter_alpha(_image, _mask, detail_range)
                _mask = tensor2pil(histogram_remap(_mask, black_point, white_point))
            else:
                _mask = tensor2pil(mask_edge_detail(_image, _mask, detail_range, black_point, white_point))

            ret_image = RGB2RGBA(orig_image, _mask.convert('L'))
            ret_images.append(pil2tensor(ret_image))
            ret_masks.append(image2mask(_mask))

        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0),)

```
