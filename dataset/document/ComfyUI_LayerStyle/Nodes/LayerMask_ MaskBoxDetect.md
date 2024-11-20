---
tags:
- Mask
---

# LayerMask: MaskBoxDetect
## Documentation
- Class name: `LayerMask: MaskBoxDetect`
- Category: `😺dzNodes/LayerMask`
- Output node: `False`

The node 'LayerMask: MaskBoxDetect' is designed to analyze and process image masks to detect and adjust bounding boxes. It supports various detection modes, including minimum bounding rectangle, maximum inscribed rectangle, and mask area detection. Additionally, it allows for adjustments in scale and position, providing flexibility in handling mask-based spatial data for image editing or analysis tasks.
## Input types
### Required
- **`mask`**
    - The 'mask' parameter represents the input image mask for which the bounding box needs to be detected. It plays a crucial role in determining the area of interest within the image.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`detect`**
    - The 'detect' parameter specifies the method to be used for bounding box detection, offering options like minimum bounding rectangle, maximum inscribed rectangle, or mask area detection. This choice affects the algorithm's approach to analyzing the mask.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`x_adjust`**
    - The 'x_adjust' parameter allows for horizontal adjustment of the detected bounding box, enabling fine-tuning of its position relative to the original mask.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`y_adjust`**
    - The 'y_adjust' parameter enables vertical adjustment of the detected bounding box, allowing for precise positioning adjustments.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`scale_adjust`**
    - The 'scale_adjust' parameter modifies the scale of the detected bounding box, offering the ability to resize it according to specific requirements.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
### Optional
## Output types
- **`box_preview`**
    - Comfy dtype: `IMAGE`
    - An image with the detected bounding box drawn over the original mask, providing a visual preview of the detection result.
    - Python dtype: `torch.Tensor`
- **`x_percent`**
    - Comfy dtype: `FLOAT`
    - The horizontal center position of the detected bounding box, expressed as a percentage of the mask's width.
    - Python dtype: `float`
- **`y_percent`**
    - Comfy dtype: `FLOAT`
    - The vertical center position of the detected bounding box, expressed as a percentage of the mask's height.
    - Python dtype: `float`
- **`width`**
    - Comfy dtype: `INT`
    - The width of the detected bounding box.
    - Python dtype: `int`
- **`height`**
    - Comfy dtype: `INT`
    - The height of the detected bounding box.
    - Python dtype: `int`
- **`x`**
    - Comfy dtype: `INT`
    - The x-coordinate of the top-left corner of the detected bounding box.
    - Python dtype: `int`
- **`y`**
    - Comfy dtype: `INT`
    - The y-coordinate of the top-left corner of the detected bounding box.
    - Python dtype: `int`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class MaskBoxDetect:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        detect_mode = ['min_bounding_rect', 'max_inscribed_rect', 'mask_area']
        return {
            "required": {
                "mask": ("MASK", ),
                "detect": (detect_mode,),  # 探测类型：最小外接矩形/最大内接矩形
                "x_adjust": ("INT", {"default": 0, "min": -9999, "max": 9999, "step": 1}),  # x轴修正
                "y_adjust": ("INT", {"default": 0, "min": -9999, "max": 9999, "step": 1}),  # y轴修正
                "scale_adjust": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100, "step": 0.01}), # 比例修正
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "FLOAT", "INT", "INT", "INT", "INT",)
    RETURN_NAMES = ("box_preview", "x_percent", "y_percent", "width", "height", "x", "y",)
    FUNCTION = 'mask_box_detect'
    CATEGORY = '😺dzNodes/LayerMask'

    def mask_box_detect(self,mask, detect, x_adjust, y_adjust, scale_adjust):

        if mask.dim() == 2:
            mask = torch.unsqueeze(mask, 0)

        if mask.shape[0] > 0:
            mask = torch.unsqueeze(mask[0], 0)

        _mask = mask2image(mask).convert('RGB')

        _mask = gaussian_blur(_mask, 20).convert('L')
        x = 0
        y = 0
        width = 0
        height = 0

        if detect == "min_bounding_rect":
            (x, y, width, height) = min_bounding_rect(_mask)
        elif detect == "max_inscribed_rect":
            (x, y, width, height) = max_inscribed_rect(_mask)
        else:
            (x, y, width, height) = mask_area(_mask)
        log(f"{NODE_NAME}: Box detected. x={x},y={y},width={width},height={height}")
        _width = width
        _height = height
        if scale_adjust != 1.0:
            _width = int(width * scale_adjust)
            _height = int(height * scale_adjust)
            x = x - int((_width - width) / 2)
            y = y - int((_height - height) / 2)
        x += x_adjust
        y += y_adjust
        x_percent = (x + _width / 2) / _mask.width * 100
        y_percent = (y + _height / 2) / _mask.height * 100
        preview_image = tensor2pil(mask).convert('RGB')
        preview_image = draw_rect(preview_image, x - x_adjust, y - y_adjust, width, height, line_color="#F00000", line_width=int(preview_image.height / 60))
        preview_image = draw_rect(preview_image, x, y, width, height, line_color="#00F000", line_width=int(preview_image.height / 40))
        log(f"{NODE_NAME} Processed.", message_type='finish')
        return ( pil2tensor(preview_image), round(x_percent, 2), round(y_percent, 2), _width, _height, x, y,)

```
