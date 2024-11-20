---
tags:
- Color
- ColorMatch
---

# Color Match Image
## Documentation
- Class name: `ColorMatchImage`
- Category: `image/filters`
- Output node: `False`

The ColorMatchImage node is designed for advanced image processing, specifically focusing on matching the color distribution of one image to another. This process is essential for tasks that require the integration of images from different sources to ensure visual consistency across them. The node offers options for selecting the type and intensity of the blur applied during the color matching process, allowing for flexible adaptation to various image characteristics.
## Input types
### Required
- **`images`**
    - The input images whose color distribution needs to be matched to the reference image. This parameter is crucial for defining the source images on which color matching will be performed.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`reference`**
    - The reference image to which the color distribution of the input images will be matched. It serves as the target for color matching, guiding the adjustment process.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`blur_type`**
    - Specifies the type of blur to be applied during the color matching process, such as Gaussian blur or guided filter, affecting the smoothness and detail preservation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`blur_size`**
    - The size of the blur filter to be applied, determining the extent of blurring and its impact on the color matching process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`factor`**
    - A factor controlling the intensity of the color matching effect, allowing for fine-tuning the balance between the original and matched colors.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output image with its color distribution matched to the reference image, integrating seamlessly with the target visual context.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ColorMatchImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
                "reference": ("IMAGE", ),
                "blur_type": (["blur", "guidedFilter"],),
                "blur_size": ("INT", {"default": 0, "min": 0, "max": 1023}),
                "factor": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01,  "round": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "batch_normalize"

    CATEGORY = "image/filters"

    def batch_normalize(self, images, reference, blur_type, blur_size, factor):
        t = images.detach().clone() + 0.1
        ref = reference.detach().clone() + 0.1
        
        if ref.shape[0] < t.shape[0]:
            ref = ref[0].unsqueeze(0).repeat(t.shape[0], 1, 1, 1)
        
        if blur_size == 0:
            mean = torch.mean(t, (1,2), keepdim=True)
            mean_ref = torch.mean(ref, (1,2), keepdim=True)
            
            for i in range(t.shape[0]):
                for c in range(3):
                    t[i,:,:,c] /= mean[i,0,0,c]
                    t[i,:,:,c] *= mean_ref[i,0,0,c]
        else:
            d = blur_size * 2 + 1
            
            if blur_type == "blur":
                blurred = cv_blur_tensor(t, d, d)
                blurred_ref = cv_blur_tensor(ref, d, d)
            elif blur_type == "guidedFilter":
                blurred = guided_filter_tensor(t, t, d, 0.01)
                blurred_ref = guided_filter_tensor(ref, ref, d, 0.01)
            
            for i in range(t.shape[0]):
                for c in range(3):
                    t[i,:,:,c] /= blurred[i,:,:,c]
                    t[i,:,:,c] *= blurred_ref[i,:,:,c]
        
        t = t - 0.1
        torch.clamp(torch.lerp(images, t, factor), 0, 1)
        return (t,)

```
