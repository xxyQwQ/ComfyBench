---
tags:
- Mask
---

# LayerMask: BiRefNetUltra
## Documentation
- Class name: `LayerMask: BiRefNetUltra`
- Category: `😺dzNodes/LayerMask`
- Output node: `False`

The BiRefNetUltra node is designed for advanced image processing, specifically focusing on background removal and refinement of layer masks. It leverages the BiRefNet architecture to achieve high-quality segmentation and mask generation, catering to applications requiring precise foreground-background separation.
## Input types
### Required
- **`image`**
    - The primary image input for processing, which the node uses to generate and refine masks based on the background and foreground segmentation.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`detail_method`**
    - Specifies the method used for enhancing the detail of the mask, including options like 'VITMatte', 'VITMatte(local)', 'PyMatting', and 'GuidedFilter', to adjust the mask's precision and quality.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`detail_erode`**
    - Determines the erosion level applied to the mask, affecting the mask's edge by reducing its size to remove small, unwanted foreground elements.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`detail_dilate`**
    - Determines the dilation level applied to the mask, affecting the mask's edge by expanding its size to include additional foreground elements that might have been missed.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`black_point`**
    - Sets the black point for histogram remapping, influencing the darkest areas of the mask to refine shadow details and contrast.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`white_point`**
    - Sets the white point for histogram remapping, influencing the brightest areas of the mask to enhance highlights and details.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`process_detail`**
    - A boolean flag that indicates whether to apply detailed processing to the mask, affecting the overall quality and precision of the mask refinement.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`device`**
    - Specifies the computing device ('cuda' or 'cpu') used for processing, affecting performance and efficiency.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`max_megapixels`**
    - Limits the maximum size of the image in megapixels to optimize processing time and resource usage, affecting the resolution of the processed image and mask.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
### Optional
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The processed image with refined foreground-background separation, suitable for further use or visualization.
    - Python dtype: `torch.Tensor`
- **`mask`**
    - Comfy dtype: `MASK`
    - The generated mask that highlights the foreground by separating it from the background, ready for use in compositing or further processing.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class BiRefNetUltra:

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
    FUNCTION = "birefnet_ultra"
    CATEGORY = '😺dzNodes/LayerMask'
  
    def birefnet_ultra(self, image, detail_method, detail_erode, detail_dilate,
                       black_point, white_point, process_detail, device, max_megapixels):
        ret_images = []
        ret_masks = []

        if detail_method == 'VITMatte(local)':
            local_files_only = True
        else:
            local_files_only = False

        from .birefnet_func import BiRefNetRemoveBackground
        birefnetrmbg = BiRefNetRemoveBackground()

        for i in image:
            i = torch.unsqueeze(i, 0)
            orig_image = tensor2pil(i).convert('RGB')

            _mask = birefnetrmbg.generate_mask(orig_image)
            _mask = image2mask(_mask)

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
                _mask = tensor2pil(_mask)

            ret_image = RGB2RGBA(orig_image, _mask.convert('L'))
            ret_images.append(pil2tensor(ret_image))
            ret_masks.append(image2mask(_mask))

        log(f"{NODE_NAME} Processed {len(ret_masks)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0),)

```
