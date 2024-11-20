---
tags:
- Image
- Segmentation
---

# LayerMask: SegmentAnythingUltra V2
## Documentation
- Class name: `LayerMask: SegmentAnythingUltra V2`
- Category: `😺dzNodes/LayerMask`
- Output node: `False`

This node specializes in segmenting various objects within an image, leveraging advanced algorithms to accurately distinguish and isolate different elements. It's designed to enhance layer-based editing by providing precise segmentation capabilities, making it a valuable tool for complex image manipulation tasks.
## Input types
### Required
- **`image`**
    - The input image to be segmented. It serves as the primary source for the segmentation process, determining the visual content the algorithm will analyze and segment.
    - Comfy dtype: `IMAGE`
    - Python dtype: `PIL.Image or torch.Tensor`
- **`sam_model`**
    - Specifies the SAM model to be used for segmentation, influencing the algorithm's approach to identifying and isolating elements within the image.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`grounding_dino_model`**
    - Determines the grounding DINO model to be utilized, affecting how the algorithm interprets and segments the image based on learned patterns and features.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`threshold`**
    - Sets the threshold for segmentation, defining the sensitivity of the algorithm in distinguishing between different elements within the image.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`detail_method`**
    - Chooses the method for enhancing detail in the segmentation, allowing for finer control over the clarity and precision of the segmented elements.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`detail_erode`**
    - Specifies the amount of erosion to apply to the details in the segmentation, helping to refine the edges and boundaries of segmented elements.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`detail_dilate`**
    - Determines the amount of dilation for the details, enhancing the visibility and prominence of segmented elements.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`black_point`**
    - Sets the black point for the segmentation, adjusting the darkest areas of the image for better contrast and visibility of elements.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`white_point`**
    - Defines the white point, modifying the brightest areas of the image to improve the segmentation's overall appearance and clarity.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`process_detail`**
    - Enables or disables the processing of fine details in the segmentation, allowing users to choose between a more general or a more detailed segmentation result.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`prompt`**
    - Provides a textual prompt to guide the segmentation process, offering contextual clues or specific instructions to the algorithm.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`device`**
    - Specifies the computational device (CPU or GPU) on which the segmentation will be performed, affecting the performance and speed of the process.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`max_megapixels`**
    - Limits the maximum resolution of the image to be processed, balancing detail against processing time and resource usage.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
### Optional
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output image after segmentation, which may include modifications or enhancements applied during the process.
    - Python dtype: `PIL.Image or torch.Tensor`
- **`mask`**
    - Comfy dtype: `MASK`
    - The generated mask corresponding to the segmented areas of the input image, useful for further editing or processing steps.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class SegmentAnythingUltraV2:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):

        method_list = ['VITMatte', 'VITMatte(local)', 'PyMatting', 'GuidedFilter', ]
        device_list = ['cuda','cpu']
        return {
            "required": {
                "image": ("IMAGE",),
                "sam_model": (list_sam_model(), ),
                "grounding_dino_model": (list_groundingdino_model(),),
                "threshold": ("FLOAT", {"default": 0.3, "min": 0, "max": 1.0, "step": 0.01}),
                "detail_method": (method_list,),
                "detail_erode": ("INT", {"default": 6, "min": 1, "max": 255, "step": 1}),
                "detail_dilate": ("INT", {"default": 6, "min": 1, "max": 255, "step": 1}),
                "black_point": ("FLOAT", {"default": 0.15, "min": 0.01, "max": 0.98, "step": 0.01, "display": "slider"}),
                "white_point": ("FLOAT", {"default": 0.99, "min": 0.02, "max": 0.99, "step": 0.01, "display": "slider"}),
                "process_detail": ("BOOLEAN", {"default": True}),
                "prompt": ("STRING", {"default": "subject"}),
                "device": (device_list,),
                "max_megapixels": ("FLOAT", {"default": 2.0, "min": 1, "max": 999, "step": 0.1}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", )
    RETURN_NAMES = ("image", "mask", )
    FUNCTION = "segment_anything_ultra_v2"
    CATEGORY = '😺dzNodes/LayerMask'

    def segment_anything_ultra_v2(self, image, sam_model, grounding_dino_model, threshold,
                                  detail_method, detail_erode, detail_dilate,
                                  black_point, white_point, process_detail, prompt,
                                  device, max_megapixels
                                  ):
        global SAM_MODEL
        global DINO_MODEL
        global previous_sam_model
        global previous_dino_model

        if detail_method == 'VITMatte(local)':
            local_files_only = True
        else:
            local_files_only = False

        if previous_sam_model != sam_model:
            SAM_MODEL = load_sam_model(sam_model)
            previous_sam_model = sam_model
        if previous_dino_model != grounding_dino_model:
            DINO_MODEL = load_groundingdino_model(grounding_dino_model)
            previous_dino_model = grounding_dino_model
        ret_images = []
        ret_masks = []

        for i in image:
            i = torch.unsqueeze(i, 0)
            i = pil2tensor(tensor2pil(i).convert('RGB'))
            _image = tensor2pil(i).convert('RGBA')
            boxes = groundingdino_predict(DINO_MODEL, _image, prompt, threshold)
            if boxes.shape[0] == 0:
                break
            (_, _mask) = sam_segment(SAM_MODEL, _image, boxes)
            _mask = _mask[0]
            detail_range = detail_erode + detail_dilate
            if process_detail:
                if detail_method == 'GuidedFilter':
                    _mask = guided_filter_alpha(i, _mask, detail_range // 6 + 1)
                    _mask = tensor2pil(histogram_remap(_mask, black_point, white_point))
                elif detail_method == 'PyMatting':
                    _mask = tensor2pil(mask_edge_detail(i, _mask, detail_range // 8 + 1, black_point, white_point))
                else:
                    _trimap = generate_VITMatte_trimap(_mask, detail_erode, detail_dilate)
                    _mask = generate_VITMatte(_image, _trimap, local_files_only=local_files_only, device=device, max_megapixels=max_megapixels)
                    _mask = tensor2pil(histogram_remap(pil2tensor(_mask), black_point, white_point))
            else:
                _mask = mask2image(_mask)
            _image = RGB2RGBA(tensor2pil(i).convert('RGB'), _mask.convert('L'))

            ret_images.append(pil2tensor(_image))
            ret_masks.append(image2mask(_mask))
        if len(ret_masks) == 0:
            _, height, width, _ = image.size()
            empty_mask = torch.zeros((1, height, width), dtype=torch.uint8, device="cpu")
            return (empty_mask, empty_mask)

        log(f"{NODE_NAME} Processed {len(ret_masks)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0),)

```
