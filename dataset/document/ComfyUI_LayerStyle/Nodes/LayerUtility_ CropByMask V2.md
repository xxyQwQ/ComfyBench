---
tags:
- Crop
- Image
- ImageTransformation
---

# LayerUtility: CropByMask V2
## Documentation
- Class name: `LayerUtility: CropByMask V2`
- Category: `😺dzNodes/LayerUtility`
- Output node: `False`

This node provides advanced image cropping capabilities based on a mask. It allows for dynamic cropping of images by applying a mask to identify the relevant area, with options for adjusting the crop area through parameters such as detection method, border reserves, and rounding to multiples. It's designed to offer flexibility in image processing tasks where precise control over the cropping area is required.
## Input types
### Required
- **`image`**
    - The image to be cropped. This parameter is crucial as it provides the visual content that will be processed and cropped according to the mask and other specified parameters.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`mask`**
    - The mask used to determine the cropping area. This parameter is essential for identifying the relevant portion of the image to be cropped, based on the mask's shape and area.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`invert_mask`**
    - A boolean flag indicating whether the mask should be inverted before processing. This affects which part of the image is considered relevant for cropping, allowing for greater flexibility in handling different types of masks.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`detect`**
    - Specifies the method used to detect the cropping area within the mask. This can include methods like 'min_bounding_rect' or 'max_inscribed_rect', influencing how the crop area is calculated.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`top_reserve`**
    - The amount of padding to reserve at the top of the crop area. This allows for fine-tuning of the crop boundary, ensuring that important image details are not lost.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`bottom_reserve`**
    - The amount of padding to reserve at the bottom of the crop area. Similar to top_reserve, this parameter helps in adjusting the crop boundary to include essential image details.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`left_reserve`**
    - The amount of padding to reserve on the left side of the crop area. It enables precise control over the crop boundary, ensuring the inclusion of important image elements.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`right_reserve`**
    - The amount of padding to reserve on the right side of the crop area. This parameter is used to adjust the crop boundary for including necessary image details.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`round_to_multiple`**
    - An optional parameter that rounds the dimensions of the crop area to the nearest multiple of a specified number. This is useful for ensuring that the cropped image meets certain size requirements.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
### Optional
- **`crop_box`**
    - An optional parameter specifying a predefined crop box. If provided, the cropping operation will use these coordinates instead of calculating them based on the mask and detection method.
    - Comfy dtype: `BOX`
    - Python dtype: `Tuple[int, int, int, int]`
## Output types
- **`croped_image`**
    - Comfy dtype: `IMAGE`
    - The image resulting from the cropping operation. This output is essential for visualizing the effect of the applied mask and cropping parameters on the original image.
    - Python dtype: `torch.Tensor`
- **`croped_mask`**
    - Comfy dtype: `MASK`
    - The mask corresponding to the cropped area of the image. This output is crucial for further processing or analysis that requires the mask of the cropped image.
    - Python dtype: `torch.Tensor`
- **`crop_box`**
    - Comfy dtype: `BOX`
    - The coordinates of the crop box determined based on the mask and additional parameters. This output is crucial for understanding the exact area of the image that has been cropped.
    - Python dtype: `Tuple[int, int, int, int]`
- **`box_preview`**
    - Comfy dtype: `IMAGE`
    - A preview image showing the crop box applied on the original image. This is useful for visual verification of the cropping area before proceeding with further image processing steps.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class CropByMaskV2:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        detect_mode = ['mask_area', 'min_bounding_rect', 'max_inscribed_rect']
        multiple_list = ['8', '16', '32', '64', 'None']
        return {
            "required": {
                "image": ("IMAGE", ),  #
                "mask": ("MASK",),
                "invert_mask": ("BOOLEAN", {"default": False}),  # 反转mask#
                "detect": (detect_mode,),
                "top_reserve": ("INT", {"default": 20, "min": -9999, "max": 9999, "step": 1}),
                "bottom_reserve": ("INT", {"default": 20, "min": -9999, "max": 9999, "step": 1}),
                "left_reserve": ("INT", {"default": 20, "min": -9999, "max": 9999, "step": 1}),
                "right_reserve": ("INT", {"default": 20, "min": -9999, "max": 9999, "step": 1}),
                "round_to_multiple": (multiple_list,),
            },
            "optional": {
                "crop_box": ("BOX",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "BOX", "IMAGE",)
    RETURN_NAMES = ("croped_image", "croped_mask", "crop_box", "box_preview")
    FUNCTION = 'crop_by_mask_v2'
    CATEGORY = '😺dzNodes/LayerUtility'

    def crop_by_mask_v2(self, image, mask, invert_mask, detect,
                     top_reserve, bottom_reserve,
                     left_reserve, right_reserve, round_to_multiple,
                     crop_box=None
                     ):

        ret_images = []
        ret_masks = []
        l_images = []
        l_masks = []

        for l in image:
            l_images.append(torch.unsqueeze(l, 0))
        if mask.dim() == 2:
            mask = torch.unsqueeze(mask, 0)
        # 如果有多张mask输入，使用第一张
        if mask.shape[0] > 1:
            log(f"Warning: Multiple mask inputs, using the first.", message_type='warning')
            mask = torch.unsqueeze(mask[0], 0)
        if invert_mask:
            mask = 1 - mask
        l_masks.append(tensor2pil(torch.unsqueeze(mask, 0)).convert('L'))

        _mask = mask2image(mask)
        preview_image = tensor2pil(mask).convert('RGB')
        if crop_box is None:
            bluredmask = gaussian_blur(_mask, 20).convert('L')
            x = 0
            y = 0
            width = 0
            height = 0
            if detect == "min_bounding_rect":
                (x, y, w, h) = min_bounding_rect(bluredmask)
            elif detect == "max_inscribed_rect":
                (x, y, w, h) = max_inscribed_rect(bluredmask)
            else:
                (x, y, w, h) = mask_area(_mask)

            canvas_width, canvas_height = tensor2pil(torch.unsqueeze(image[0], 0)).convert('RGB').size
            x1 = x - left_reserve if x - left_reserve > 0 else 0
            y1 = y - top_reserve if y - top_reserve > 0 else 0
            x2 = x + w + right_reserve if x + w + right_reserve < canvas_width else canvas_width
            y2 = y + h + bottom_reserve if y + h + bottom_reserve < canvas_height else canvas_height

            if round_to_multiple != 'None':
                multiple = int(round_to_multiple)
                width = num_round_up_to_multiple(x2 - x1, multiple)
                height = num_round_up_to_multiple(y2 - y1, multiple)
                x1 = x1 - (width - (x2 - x1)) // 2
                y1 = y1 - (height - (y2 - y1)) // 2
                x2 = x1 + width
                y2 = y1 + height

            log(f"{NODE_NAME}: Box detected. x={x1},y={y1},width={width},height={height}")
            crop_box = (x1, y1, x2, y2)
            preview_image = draw_rect(preview_image, x, y, w, h, line_color="#F00000",
                                      line_width=(w + h) // 100)
        preview_image = draw_rect(preview_image, crop_box[0], crop_box[1],
                                  crop_box[2] - crop_box[0], crop_box[3] - crop_box[1],
                                  line_color="#00F000",
                                  line_width=(crop_box[2] - crop_box[0] + crop_box[3] - crop_box[1]) // 200)
        for i in range(len(l_images)):
            _canvas = tensor2pil(l_images[i]).convert('RGB')
            _mask = l_masks[0]
            ret_images.append(pil2tensor(_canvas.crop(crop_box)))
            ret_masks.append(image2mask(_mask.crop(crop_box)))

        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0), list(crop_box), pil2tensor(preview_image),)

```
