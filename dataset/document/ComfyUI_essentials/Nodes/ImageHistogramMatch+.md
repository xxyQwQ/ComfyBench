---
tags:
- Color
---

# 🔧 Image Histogram Match
## Documentation
- Class name: `ImageHistogramMatch+`
- Category: `essentials/image processing`
- Output node: `False`

This node is designed to perform histogram matching on images, a process that adjusts the pixel values of an image so its histogram matches that of a reference image. This technique is useful for color correction and achieving consistent visual styles across different images.
## Input types
### Required
- **`image`**
    - The image to be processed and adjusted to match the histogram of the reference image. It plays a pivotal role in the histogram matching operation, determining the final appearance of the adjusted image.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`reference`**
    - The reference image providing the desired histogram distribution for the histogram matching process. It establishes the target histogram characteristics for the image adjustment.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`method`**
    - Specifies the method to be used for histogram matching. This parameter allows for the selection between different histogram matching techniques, influencing the approach and potentially the quality of the result.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`factor`**
    - A blending factor that determines the degree to which the matched histogram influences the final image, allowing for partial adjustments.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`device`**
    - Indicates the computational device ('cpu', 'gpu', or 'auto') where the histogram matching process will be executed, affecting performance and resource utilization.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output image after applying histogram matching, reflecting the adjusted pixel values to match the histogram of the reference image.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ImageHistogramMatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "reference": ("IMAGE",),
                "method": (["pytorch", "skimage"],),
                "factor": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, }),
                "device": (["auto", "cpu", "gpu"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image processing"

    def execute(self, image, reference, method, factor, device):
        if "gpu" == device:
            device = comfy.model_management.get_torch_device()
        elif "auto" == device:
            device = comfy.model_management.intermediate_device()
        else:
            device = 'cpu'

        if "pytorch" in method:
            from .histogram_matching import Histogram_Matching

            image = image.permute([0, 3, 1, 2]).to(device)
            reference = reference.permute([0, 3, 1, 2]).to(device)[0].unsqueeze(0)
            image.requires_grad = True
            reference.requires_grad = True

            out = []

            for i in image:
                i = i.unsqueeze(0)
                hm = Histogram_Matching(differentiable=True)
                out.append(hm(i, reference))
            out = torch.cat(out, dim=0)
            out = factor * out + (1 - factor) * image
            out = out.permute([0, 2, 3, 1]).clamp(0, 1)
        else:
            from skimage.exposure import match_histograms

            out = torch.from_numpy(match_histograms(image.cpu().numpy(), reference.cpu().numpy(), channel_axis=3)).to(device)
            out = factor * out + (1 - factor) * image.to(device)

        return (out.to(comfy.model_management.intermediate_device()),)

```
