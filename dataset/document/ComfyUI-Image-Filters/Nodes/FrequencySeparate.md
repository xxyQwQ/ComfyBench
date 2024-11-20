# Frequency Separate
## Documentation
- Class name: `FrequencySeparate`
- Category: `image/filters`
- Output node: `False`

The FrequencySeparate node is designed for image processing, specifically to separate the high-frequency components from an image by either subtracting or dividing the low-frequency components. This operation is useful for tasks that require manipulation or analysis of an image's texture or fine details without affecting the overall structure.
## Input types
### Required
- **`original`**
    - The original image from which high-frequency components are to be extracted. It plays a crucial role in determining the texture and fine details to be separated.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`low_frequency`**
    - The low-frequency component of the image, used as a basis for extracting high-frequency details by comparison with the original image.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`mode`**
    - Determines the method of separation (either 'subtract' or 'divide'), affecting how the high-frequency components are calculated from the original image.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
- **`eps`**
    - A small epsilon value to prevent division by zero when 'divide' mode is selected, ensuring numerical stability.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`high_frequency`**
    - Comfy dtype: `IMAGE`
    - The high-frequency component of the image, representing the texture and fine details separated from the original image.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class FrequencySeparate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "original": ("IMAGE", ),
                "low_frequency": ("IMAGE", ),
                "mode": (["subtract", "divide"],),
                "eps": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 0.99, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("high_frequency",)
    FUNCTION = "filter_image"

    CATEGORY = "image/filters"

    def filter_image(self, original, low_frequency, mode, eps):
        t = original.detach().clone()
        if mode == "subtract":
            t = t - low_frequency + 0.5
        else:
            t = ((t + eps) / (low_frequency + eps)) * 0.5
        return (t,)

```
