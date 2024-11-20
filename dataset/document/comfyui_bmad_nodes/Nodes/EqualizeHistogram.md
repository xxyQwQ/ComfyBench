# EqualizeHistogram
## Documentation
- Class name: `EqualizeHistogram`
- Category: `Bmad/CV/Thresholding`
- Output node: `False`

The EqualizeHistogram node is designed to enhance the contrast of an image by equalizing its histogram. This process redistributes the intensity levels of the image, potentially improving its visual perception.
## Input types
### Required
- **`src`**
    - The source image to be processed for histogram equalization. This input is crucial as it directly influences the enhancement of the image's contrast.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output is an enhanced image with equalized histogram, aimed at improving the contrast and visual quality.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class EqualizeHistogram:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "src": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "eq"
    CATEGORY = f"{cv_category_path}/Thresholding"

    def eq(self, src):
        src = tensor2opencv(src, 1)
        eq = cv.equalizeHist(src)
        eq = cv.cvtColor(eq, cv.COLOR_GRAY2RGB)
        eq = opencv2tensor(eq)
        return (eq,)

```
