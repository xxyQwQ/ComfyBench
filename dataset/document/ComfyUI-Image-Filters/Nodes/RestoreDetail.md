---
tags:
- VisualEffects
---

# Restore Detail
## Documentation
- Class name: `RestoreDetail`
- Category: `image/filters`
- Output node: `False`

The RestoreDetail node is designed to enhance the visual quality of images by restoring details that may have been lost or obscured. It focuses on improving the clarity and sharpness of images, making them appear more vivid and lifelike.
## Input types
### Required
- **`images`**
    - The input images to be processed for detail restoration, aiming to enhance clarity and sharpness.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`detail`**
    - A secondary image or detail layer that guides the restoration process, emphasizing specific details to be enhanced or recovered.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`mode`**
    - Specifies the operation mode (e.g., 'add' or 'multiply') to combine the input image with the detail layer, affecting the enhancement effect.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`blur_type`**
    - Determines the type of blur (e.g., 'blur' or 'guidedFilter') applied as part of the detail restoration process.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`blur_size`**
    - The size of the blur filter to be applied, influencing the smoothness or sharpness of the restored details.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`factor`**
    - A scaling factor that adjusts the intensity of the detail restoration effect.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The enhanced image with restored details, reflecting the positive aspects described in the input.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class RestoreDetail:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
                "detail": ("IMAGE", ),
                "mode": (["add", "multiply"],),
                "blur_type": (["blur", "guidedFilter"],),
                "blur_size": ("INT", {"default": 1, "min": 1, "max": 1023}),
                "factor": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01,  "round": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "batch_normalize"

    CATEGORY = "image/filters"

    def batch_normalize(self, images, detail, mode, blur_type, blur_size, factor):
        t = images.detach().clone() + 0.1
        ref = detail.detach().clone() + 0.1
        
        if ref.shape[0] < t.shape[0]:
            ref = ref[0].unsqueeze(0).repeat(t.shape[0], 1, 1, 1)
        
        d = blur_size * 2 + 1
        
        if blur_type == "blur":
            blurred = cv_blur_tensor(t, d, d)
            blurred_ref = cv_blur_tensor(ref, d, d)
        elif blur_type == "guidedFilter":
            blurred = guided_filter_tensor(t, t, d, 0.01)
            blurred_ref = guided_filter_tensor(ref, ref, d, 0.01)
        
        if mode == "multiply":
            t = (ref / blurred_ref) * blurred
        else:
            t = (ref - blurred_ref) + blurred
        
        t = t - 0.1
        t = torch.clamp(torch.lerp(images, t, factor), 0, 1)
        return (t,)

```
