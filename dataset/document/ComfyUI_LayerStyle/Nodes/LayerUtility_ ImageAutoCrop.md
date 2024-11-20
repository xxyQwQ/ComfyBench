---
tags:
- Crop
- Image
- ImageTransformation
---

# LayerUtility: ImageAutoCrop
## Documentation
- Class name: `LayerUtility: ImageAutoCrop`
- Category: `😺dzNodes/LayerUtility`
- Output node: `False`

The ImageAutoCrop node is designed to automatically crop images based on certain criteria, such as the presence of a mask or specific image features. It aims to streamline the process of refining image layers by removing unnecessary or undesired parts, enhancing the focus and composition of the image.
## Input types
### Required
- **`image`**
    - Represents the input image to be auto-cropped. It is crucial for determining the area of interest and executing the crop operation.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`background_color`**
    - Specifies the background color to be used in the cropping process, particularly when adjusting the image to a new aspect ratio or filling in areas after cropping.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`aspect_ratio`**
    - Defines the desired aspect ratio for the cropped image, which can be set to custom values, original, or based on the detection of a mask.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`proportional_width`**
    - When using a custom aspect ratio, this value specifies the width proportion to calculate the new aspect ratio.
    - Comfy dtype: `INT`
    - Python dtype: `float`
- **`proportional_height`**
    - When using a custom aspect ratio, this value specifies the height proportion to calculate the new aspect ratio.
    - Comfy dtype: `INT`
    - Python dtype: `float`
- **`scale_to_longest_side`**
    - A boolean indicating whether to scale the image based on the longest side, maintaining the aspect ratio.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`longest_side`**
    - Specifies the target length of the longest side of the image when scaling is applied.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`detect`**
    - Determines the method used to detect the cropping area, such as minimum bounding rectangle, maximum inscribed rectangle, or mask area detection.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`border_reserve`**
    - The amount of border to reserve around the detected cropping area, adding space around the crop.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`ultra_detail_range`**
    - Specifies the range for enhancing detail around the edges of the mask, used in conjunction with certain matting methods.
    - Comfy dtype: `INT`
    - Python dtype: `float`
- **`matting_method`**
    - The method used for generating a mask around the subject of the image, such as 'SegmentAnything' or a simple background removal.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`sam_model`**
    - Specifies the model used for the 'SegmentAnything' matting method, affecting the quality and accuracy of the generated mask.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`grounding_dino_model`**
    - Specifies the DINO model used for grounding the segmentation, enhancing the mask's accuracy around the subject.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`sam_threshold`**
    - The threshold value for the SAM model, determining the sensitivity of the segmentation process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sam_prompt`**
    - The prompt used with the SAM model to guide the segmentation process, influencing the areas of focus.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
## Output types
- **`cropped_image`**
    - Comfy dtype: `IMAGE`
    - The result of the auto-cropping process, which is a refined version of the original image with unnecessary parts removed.
    - Python dtype: `torch.Tensor`
- **`box_preview`**
    - Comfy dtype: `IMAGE`
    - A visual representation of the cropping box applied to the original image, useful for previewing the crop area before finalization.
    - Python dtype: `torch.Tensor`
- **`cropped_mask`**
    - Comfy dtype: `MASK`
    - The corresponding mask of the cropped image, adjusted to match the new dimensions and area of the cropped image.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ImageAutoCrop:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        matting_method_list = ['RMBG 1.4', 'SegmentAnything']
        detect_mode = ['min_bounding_rect', 'max_inscribed_rect', 'mask_area']
        ratio_list = ['1:1', '3:2', '4:3', '16:9', '2:3', '3:4', '9:16', 'custom', 'detect_mask']
        return {
            "required": {
                "image": ("IMAGE", ),  #
                "background_color": ("STRING", {"default": "#FFFFFF"}),  # 背景颜色
                "aspect_ratio": (ratio_list,),
                "proportional_width": ("INT", {"default": 2, "min": 1, "max": 999, "step": 1}),
                "proportional_height": ("INT", {"default": 1, "min": 1, "max": 999, "step": 1}),
                "scale_to_longest_side": ("BOOLEAN", {"default": True}),  # 是否按长边缩放
                "longest_side": ("INT", {"default": 1024, "min": 4, "max": 999999, "step": 1}),
                "detect": (detect_mode,),
                "border_reserve": ("INT", {"default": 100, "min": -9999, "max": 9999, "step": 1}),
                "ultra_detail_range": ("INT", {"default": 0, "min": 0, "max": 256, "step": 1}),
                "matting_method": (matting_method_list,),
                "sam_model": (list_sam_model(),),
                "grounding_dino_model": (list_groundingdino_model(),),
                "sam_threshold": ("FLOAT", {"default": 0.3, "min": 0, "max": 1.0, "step": 0.01}),
                "sam_prompt": ("STRING", {"default": "subject"}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK",)
    RETURN_NAMES = ("cropped_image", "box_preview", "cropped_mask",)
    FUNCTION = 'image_auto_crop'
    CATEGORY = '😺dzNodes/LayerUtility'

    def image_auto_crop(self, image, detect, border_reserve, aspect_ratio, proportional_width, proportional_height,
                        background_color, ultra_detail_range, scale_to_longest_side, longest_side,
                        matting_method, sam_model, grounding_dino_model, sam_threshold, sam_prompt
                        ):

        ret_images = []
        ret_box_previews = []
        ret_masks = []
        input_images = []
        input_masks = []
        crop_boxs = []

        for l in image:
            input_images.append(torch.unsqueeze(l, 0))
            m = tensor2pil(l)
            if m.mode == 'RGBA':
                input_masks.append(m.split()[-1])

        if len(input_masks) > 0 and len(input_masks) != len(input_images):
            input_masks = []
            log(f"Warning, {NODE_NAME} unable align alpha to image, drop it.", message_type='warning')

        if aspect_ratio == 'custom':
            ratio = proportional_width / proportional_height
        elif aspect_ratio == 'detect_mask':
            ratio = 0
        else:
            s = aspect_ratio.split(":")
            ratio = int(s[0]) / int(s[1])
        side_limit = longest_side if scale_to_longest_side else 0

        for i in range(len(input_images)):
            _image = tensor2pil(input_images[i]).convert('RGB')
            if len(input_masks) > 0:
                _mask = input_masks[i]
            else:
                if matting_method == 'SegmentAnything':
                    sam_model = load_sam_model(sam_model)
                    dino_model = load_groundingdino_model(grounding_dino_model)
                    item = _image.convert('RGBA')
                    boxes = groundingdino_predict(dino_model, item, sam_prompt, sam_threshold)
                    (_, _mask) = sam_segment(sam_model, item, boxes)
                    _mask = mask2image(_mask[0])
                else:
                    _mask = RMBG(_image)
                if ultra_detail_range:
                    _mask = tensor2pil(mask_edge_detail(input_images[i], pil2tensor(_mask), ultra_detail_range, 0.01, 0.99))
            bluredmask = gaussian_blur(_mask, 20).convert('L')
            x = 0
            y = 0
            width = 0
            height = 0
            x_offset = 0
            y_offset = 0
            if detect == "min_bounding_rect":
                (x, y, width, height) = min_bounding_rect(bluredmask)
            elif detect == "max_inscribed_rect":
                (x, y, width, height) = max_inscribed_rect(bluredmask)
            else:
                (x, y, width, height) = mask_area(bluredmask)
            canvas_width, canvas_height = _image.size
            x1 = x - border_reserve
            y1 = y - border_reserve
            x2 = x + width + border_reserve
            y2 = y + height + border_reserve
            if x1 < 0:
                canvas_width -= x1
                x_offset = -x1
            if y1 < 0:
                canvas_height -= y1
                y_offset = -y1
            if x2 > _image.width:
                canvas_width += x2 - _image.width
            if y2 > _image.height:
                canvas_height += y2 - _image.height
            crop_box = (x1 + x_offset, y1 + y_offset, width + border_reserve*2, height + border_reserve*2)
            crop_boxs.append(crop_box)
            if len(crop_boxs) > 0:    # 批量图强制使用同一尺寸
                crop_box = crop_boxs[0]
            if aspect_ratio == 'detect_mask':
                ratio = crop_box[2] / crop_box[3]
            target_width, target_height = calculate_side_by_ratio(crop_box[2], crop_box[3], ratio,
                                                                  longest_side=side_limit)
            _canvas = Image.new('RGB', size=(canvas_width, canvas_height), color=background_color)
            _mask_canvas = Image.new('L',  size=(canvas_width, canvas_height), color='black')
            if ultra_detail_range:
                _image = pixel_spread(_image, _mask)
            _canvas.paste(_image, box=(x_offset, y_offset), mask=_mask.convert('L'))
            _mask_canvas.paste(_mask, box=(x_offset, y_offset))
            preview_image = Image.new('RGB', size=(canvas_width, canvas_height), color='gray')
            preview_image.paste(_mask, box=(x_offset, y_offset))
            preview_image = draw_rect(preview_image,
                                      crop_box[0], crop_box[1], crop_box[2], crop_box[3],
                                      line_color="#F00000", line_width=(canvas_width + canvas_height)//200)

            ret_image = _canvas.crop((crop_box[0], crop_box[1], crop_box[0]+crop_box[2], crop_box[1]+crop_box[3]))
            ret_image = fit_resize_image(ret_image, target_width, target_height,
                                         fit='letterbox', resize_sampler=Image.LANCZOS,
                                         background_color=background_color)
            ret_mask = _mask_canvas.crop((crop_box[0], crop_box[1], crop_box[0]+crop_box[2], crop_box[1]+crop_box[3]))
            ret_mask = fit_resize_image(ret_mask, target_width, target_height,
                                         fit='letterbox', resize_sampler=Image.LANCZOS,
                                         background_color="#000000")
            ret_images.append(pil2tensor(ret_image))
            ret_box_previews.append(pil2tensor(preview_image))
            ret_masks.append(image2mask(ret_mask))

        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0),
                torch.cat(ret_box_previews, dim=0),
                torch.cat(ret_masks, dim=0),
                )

```
