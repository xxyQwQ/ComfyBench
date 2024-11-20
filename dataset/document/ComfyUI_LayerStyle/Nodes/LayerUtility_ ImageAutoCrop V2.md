---
tags:
- Crop
- Image
- ImageTransformation
---

# LayerUtility: ImageAutoCrop V2
## Documentation
- Class name: `LayerUtility: ImageAutoCrop V2`
- Category: `😺dzNodes/LayerUtility`
- Output node: `False`

This node automatically crops images based on their content, enhancing focus and composition by removing unnecessary background. It utilizes advanced algorithms to detect and retain the main subject of the image while discarding the extraneous parts, providing a cleaner, more focused visual output.
## Input types
### Required
- **`image`**
    - The input image to be auto-cropped. This parameter is crucial as it determines the area of focus and the parts to be eliminated based on the content and composition of the image.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`fill_background`**
    - A boolean flag indicating whether to fill the background after cropping. If true, the area outside the cropped region will be filled with the specified background color.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`background_color`**
    - The color used to fill the background if fill_background is set to true. This parameter allows customization of the background color post-cropping.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`aspect_ratio`**
    - Specifies the desired aspect ratio for the cropped image. This can be set to 'custom', 'original', 'detect_mask', or a specific ratio like '16:9', guiding the cropping process to maintain the specified aspect ratio.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`proportional_width`**
    - When aspect_ratio is set to 'custom', this parameter defines the width component of the custom aspect ratio.
    - Comfy dtype: `INT`
    - Python dtype: `float`
- **`proportional_height`**
    - When aspect_ratio is set to 'custom', this parameter defines the height component of the custom aspect ratio.
    - Comfy dtype: `INT`
    - Python dtype: `float`
- **`scale_to_side`**
    - A boolean flag that, when true, scales the image to fit the longest side to the specified scale_to_length, maintaining the aspect ratio.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `bool`
- **`scale_to_length`**
    - Specifies the length to which the longest side of the image should be scaled when scale_to_side is true.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`detect`**
    - Determines the method used to detect the cropping area. Options include 'min_bounding_rect', 'max_inscribed_rect', and 'mask_area', each affecting how the crop area is calculated.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`border_reserve`**
    - An additional border added around the detected crop area to ensure important parts of the image are not cut off. Specified in pixels.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`ultra_detail_range`**
    - Specifies the range for enhancing detail in the cropped area, useful for fine-tuning the focus on specific parts of the image.
    - Comfy dtype: `INT`
    - Python dtype: `float`
- **`matting_method`**
    - The method used for image matting, which helps in separating the foreground from the background. Options include 'SegmentAnything' and other matting techniques.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`sam_model`**
    - Specifies the model used for the 'SegmentAnything' matting method, affecting the quality and accuracy of the foreground extraction.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`grounding_dino_model`**
    - Specifies the DINO model used for grounding, which assists in better foreground segmentation when using 'SegmentAnything'.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`sam_threshold`**
    - The threshold value for the 'SegmentAnything' method, determining the sensitivity of the segmentation process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sam_prompt`**
    - The prompt used for the 'SegmentAnything' method, guiding the model in understanding what part of the image to focus on during segmentation.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
- **`mask`**
    - An optional mask that can be provided to guide the cropping process. It helps in specifying the areas of the image that should be retained or removed, enhancing the accuracy of the auto-cropping algorithm.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
## Output types
- **`cropped_image`**
    - Comfy dtype: `IMAGE`
    - The result of the auto-cropping process, which is a version of the original image with unnecessary background removed, focusing on the main subject.
    - Python dtype: `torch.Tensor`
- **`box_preview`**
    - Comfy dtype: `IMAGE`
    - A visual preview of the bounding box used for cropping the image. It helps in understanding how the image was cropped and the area that was focused on.
    - Python dtype: `torch.Tensor`
- **`cropped_mask`**
    - Comfy dtype: `MASK`
    - The mask corresponding to the cropped area of the image. It indicates the parts of the image that were retained after the cropping process.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ImageAutoCropV2:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        matting_method_list = ['RMBG 1.4', 'SegmentAnything']
        detect_mode = ['min_bounding_rect', 'max_inscribed_rect', 'mask_area']
        ratio_list = ['1:1', '3:2', '4:3', '16:9', '2:3', '3:4', '9:16', 'custom', 'detect_mask', 'original']
        scale_to_side_list = ['None', 'longest', 'shortest', 'width', 'height']
        return {
            "required": {
                "image": ("IMAGE", ),  #
                "fill_background": ("BOOLEAN", {"default": True}),  # 是否填充背景
                "background_color": ("STRING", {"default": "#FFFFFF"}),  # 背景颜色
                "aspect_ratio": (ratio_list,),
                "proportional_width": ("INT", {"default": 1, "min": 1, "max": 999, "step": 1}),
                "proportional_height": ("INT", {"default": 1, "min": 1, "max": 999, "step": 1}),
                "scale_to_side": (scale_to_side_list,),  # 是否按长边缩放
                "scale_to_length": ("INT", {"default": 1024, "min": 4, "max": 999999, "step": 1}),
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
                "mask": ("MASK",),  #
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK",)
    RETURN_NAMES = ("cropped_image", "box_preview", "cropped_mask",)
    FUNCTION = 'image_auto_crop_v2'
    CATEGORY = '😺dzNodes/LayerUtility'

    def image_auto_crop_v2(self, image, fill_background, background_color, aspect_ratio,
                        proportional_width, proportional_height,
                        scale_to_side, scale_to_length, detect, border_reserve,
                        ultra_detail_range, matting_method,
                        sam_model, grounding_dino_model, sam_threshold, sam_prompt,
                        mask=None,
                        ):

        ret_images = []
        ret_box_previews = []
        ret_masks = []
        input_images = []
        input_masks = []
        crop_boxs = []

        global SAM_MODEL
        global DINO_MODEL
        global previous_sam_model
        global previous_dino_model

        for l in image:
            input_images.append(torch.unsqueeze(l, 0))
            m = tensor2pil(l)
            if m.mode == 'RGBA':
                input_masks.append(m.split()[-1])
        if mask is not None:
            if mask.dim() == 2:
                mask = torch.unsqueeze(mask, 0)
            input_masks = []
            for m in mask:
                input_masks.append(tensor2pil(torch.unsqueeze(m, 0)).convert('L'))

        if len(input_masks) > 0 and len(input_masks) != len(input_images):
            input_masks = []
            log(f"Warning, {NODE_NAME} unable align alpha to image, drop it.", message_type='warning')

        fit = 'letterbox'
        if aspect_ratio == 'custom':
            ratio = proportional_width / proportional_height
        elif aspect_ratio == 'original':
            _image = tensor2pil(input_images[0])
            ratio = _image.width / _image.height
        elif aspect_ratio == 'detect_mask':
            ratio = 0
            fit = 'fill'
        else:
            s = aspect_ratio.split(":")
            ratio = int(s[0]) / int(s[1])

        for i in range(len(input_images)):
            _image = tensor2pil(input_images[i]).convert('RGB')

            if len(input_masks) > 0:
                _mask = input_masks[i]
            else:
                if matting_method == 'SegmentAnything':
                    if previous_sam_model != sam_model:
                        SAM_MODEL = load_sam_model(sam_model)
                        previous_sam_model = sam_model
                    if previous_dino_model != grounding_dino_model:
                        DINO_MODEL = load_groundingdino_model(grounding_dino_model)
                        previous_dino_model = grounding_dino_model
                    item = _image.convert('RGBA')
                    boxes = groundingdino_predict(DINO_MODEL, item, sam_prompt, sam_threshold)
                    (_, _mask) = sam_segment(SAM_MODEL, item, boxes)
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
                if fill_background:
                    canvas_width -= x1
                    x_offset = -x1
                else:
                    x1 = 0
            if y1 < 0:
                if fill_background:
                    canvas_height -= y1
                    y_offset = -y1
                else:
                    y1 = 0
            if x2 > _image.width:
                if fill_background:
                    canvas_width += x2 - _image.width
                else:
                    x2 = _image.width
            if y2 > _image.height:
                if fill_background:
                    canvas_height += y2 - _image.height
                else:
                    y2 = _image.height

            if fill_background:
                crop_box = (x1 + x_offset, y1 + y_offset, width + border_reserve*2, height + border_reserve*2)
            else:
                crop_box = (x1, y1, x2 - x1, y2 - y1)
            crop_boxs.append(crop_box)
            if len(crop_boxs) > 0:    # 批量图强制使用同一尺寸
                crop_box = crop_boxs[0]

            orig_width =  crop_box[2]
            orig_height = crop_box[3]
            if aspect_ratio == 'detect_mask':
                ratio = orig_width / orig_height

            # calculate target width and height
            if orig_width > orig_height:
                if scale_to_side == 'longest':
                    target_width = scale_to_length
                    target_height = int(target_width / ratio)
                elif scale_to_side == 'shortest':
                    target_height = scale_to_length
                    target_width = int(target_height * ratio)
                elif scale_to_side == 'width':
                    target_width = scale_to_length
                    target_height = int(target_width / ratio)
                elif scale_to_side == 'height':
                    target_height = scale_to_length
                    target_width = int(target_height * ratio)
                else:
                    target_width = orig_width
                    target_height = int(target_width / ratio)
            else:
                if scale_to_side == 'longest':
                    target_height = scale_to_length
                    target_width = int(target_height * ratio)
                elif scale_to_side == 'shortest':
                    target_width = scale_to_length
                    target_height = int(target_width / ratio)
                elif scale_to_side == 'width':
                    target_width = scale_to_length
                    target_height = int(target_width / ratio)
                elif scale_to_side == 'height':
                    target_height = scale_to_length
                    target_width = int(target_height * ratio)
                else:
                    target_height = orig_height
                    target_width = int(target_height * ratio)

            _canvas = Image.new('RGB', size=(canvas_width, canvas_height), color=background_color)
            _mask_canvas = Image.new('L',  size=(canvas_width, canvas_height), color='black')
            if ultra_detail_range:
                _image = pixel_spread(_image, _mask)
            if fill_background:
                _canvas.paste(_image, box=(x_offset, y_offset), mask=_mask.convert('L'))
            else:
                _canvas.paste(_image, box=(x_offset, y_offset))
            _mask_canvas.paste(_mask, box=(x_offset, y_offset))
            preview_image = Image.new('RGB', size=(canvas_width, canvas_height), color='gray')
            preview_image.paste(_mask, box=(x_offset, y_offset))
            preview_image = draw_rect(preview_image,
                                      crop_box[0], crop_box[1], crop_box[2], crop_box[3],
                                      line_color="#F00000", line_width=(canvas_width + canvas_height)//200)

            ret_image = _canvas.crop((crop_box[0], crop_box[1], crop_box[0]+crop_box[2], crop_box[1]+crop_box[3]))
            ret_image = fit_resize_image(ret_image, target_width, target_height,
                                         fit=fit, resize_sampler=Image.LANCZOS,
                                         background_color=background_color)
            ret_mask = _mask_canvas.crop((crop_box[0], crop_box[1], crop_box[0]+crop_box[2], crop_box[1]+crop_box[3]))
            ret_mask = fit_resize_image(ret_mask, target_width, target_height,
                                         fit=fit, resize_sampler=Image.LANCZOS,
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
