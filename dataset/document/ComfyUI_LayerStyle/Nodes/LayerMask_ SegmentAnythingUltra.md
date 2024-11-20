---
tags:
- Segmentation
---

# LayerMask: SegmentAnythingUltra
## Documentation
- Class name: `LayerMask: SegmentAnythingUltra`
- Category: `😺dzNodes/LayerMask`
- Output node: `False`

This node specializes in segmenting various objects within an image, leveraging advanced algorithms to accurately distinguish and isolate different elements. It's designed to enhance layer-based editing by providing precise mask generation for any object, facilitating complex image manipulation tasks.
## Input types
### Required
- **`image`**
    - The input image to be segmented. This parameter is crucial as it directly influences the segmentation accuracy and the quality of the generated mask.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`sam_model`**
    - Specifies the SAM model used for segmentation, impacting the segmentation's effectiveness and accuracy.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`grounding_dino_model`**
    - Defines the DINO model used for object detection within the image, crucial for identifying objects to segment.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`threshold`**
    - The confidence threshold for object detection, determining which objects are considered for segmentation.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`detail_range`**
    - A parameter influencing the level of detail in the segmentation process, affecting the mask's precision around object edges.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`black_point`**
    - Sets the black point for histogram remapping, affecting the contrast and detail of the segmented mask.
    - Comfy dtype: `FLOAT`
    - Python dtype: `int`
- **`white_point`**
    - Defines the white point for histogram remapping, adjusting the brightness and detail visibility in the mask.
    - Comfy dtype: `FLOAT`
    - Python dtype: `int`
- **`process_detail`**
    - A boolean indicating whether to apply detailed processing to the mask, enhancing edge definition and overall mask quality.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`prompt`**
    - The text prompt used for guiding the object detection process, influencing the specificity and accuracy of detected objects.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The segmented image with applied mask, showcasing the isolated objects against the original background. This output is crucial for visualizing the segmentation results and for further processing or editing.
    - Python dtype: `torch.Tensor`
- **`mask`**
    - Comfy dtype: `MASK`
    - The output mask generated from the input image, representing the segmented areas. This mask is essential for further image editing and manipulation, providing a clear distinction between different objects within the image.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class SegmentAnythingUltra:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):

        return {
            "required": {
                "image": ("IMAGE",),
                "sam_model": (list_sam_model(), ),
                "grounding_dino_model": (list_groundingdino_model(),),
                "threshold": ("FLOAT", {"default": 0.3, "min": 0, "max": 1.0, "step": 0.01}),
                "detail_range": ("INT", {"default": 16, "min": 1, "max": 256, "step": 1}),
                "black_point": ("FLOAT", {"default": 0.15, "min": 0.01, "max": 0.98, "step": 0.01}),
                "white_point": ("FLOAT", {"default": 0.99, "min": 0.02, "max": 0.99, "step": 0.01}),
                "process_detail": ("BOOLEAN", {"default": True}),
                "prompt": ("STRING", {"default": "subject"}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", )
    RETURN_NAMES = ("image", "mask", )
    FUNCTION = "segment_anything_ultra"
    CATEGORY = '😺dzNodes/LayerMask'

    def segment_anything_ultra(self, image, sam_model, grounding_dino_model, threshold,
                               detail_range, black_point, white_point, process_detail,
                               prompt, ):
        global SAM_MODEL
        global DINO_MODEL
        if SAM_MODEL is None: SAM_MODEL = load_sam_model(sam_model)
        if DINO_MODEL is None: DINO_MODEL = load_groundingdino_model(grounding_dino_model)
        ret_images = []
        ret_masks = []

        for i in image:
            i = torch.unsqueeze(i, 0)
            i = pil2tensor(tensor2pil(i).convert('RGB'))
            item = tensor2pil(i).convert('RGBA')
            boxes = groundingdino_predict(DINO_MODEL, item, prompt, threshold)
            if boxes.shape[0] == 0:
                break
            (_, _mask) = sam_segment(SAM_MODEL, item, boxes)
            _mask = _mask[0]
            if process_detail:
                _mask = tensor2pil(mask_edge_detail(i, _mask, detail_range, black_point, white_point))
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
