---
tags:
- Color
- ColorMatch
---

# 🔧 Image Color Match
## Documentation
- Class name: `ImageColorMatch+`
- Category: `essentials/image processing`
- Output node: `False`

This node is designed to adjust the colors of an image to match those of a reference image, utilizing color space conversions and histogram matching techniques to achieve a visually harmonious result. It leverages advanced image processing algorithms to ensure that the output image closely mirrors the color distribution and tone of the target, making it ideal for applications requiring color consistency across different images.
## Input types
### Required
- **`image`**
    - The input image to be color-matched. This image will undergo color space conversion and histogram matching to align its color profile with that of the reference image.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`reference`**
    - The reference image whose color profile is to be matched. This image serves as the target for the color matching process, guiding the adjustments made to the input image.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`color_space`**
    - The color space to be used for the color matching process. This determines how the colors are interpreted and adjusted during the matching operation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`factor`**
    - A factor that blends the matched image with the original input image to control the intensity of the color matching.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`device`**
    - Specifies the device (CPU, GPU, or auto) on which the color matching operation is performed.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output image after color matching. This image has been adjusted to closely match the color profile of the reference image, ensuring color consistency.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ImageColorMatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "reference": ("IMAGE",),
                "color_space": (["LAB", "YCbCr", "RGB", "LUV", "YUV", "XYZ"],),
                "factor": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, }),
                "device": (["auto", "cpu", "gpu"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image processing"

    def execute(self, image, reference, color_space, factor, device):
        import kornia

        if "gpu" == device:
            device = comfy.model_management.get_torch_device()
        elif "auto" == device:
            device = comfy.model_management.intermediate_device()
        else:
            device = 'cpu'

        image = image.permute([0, 3, 1, 2]).to(device)
        reference = reference.permute([0, 3, 1, 2]).to(device)

        if "LAB" == color_space:
            image = kornia.color.rgb_to_lab(image)
            reference = kornia.color.rgb_to_lab(reference)
        elif "YCbCr" == color_space:
            image = kornia.color.rgb_to_ycbcr(image)
            reference = kornia.color.rgb_to_ycbcr(reference)
        elif "LUV" == color_space:
            image = kornia.color.rgb_to_luv(image)
            reference = kornia.color.rgb_to_luv(reference)
        elif "YUV" == color_space:
            image = kornia.color.rgb_to_yuv(image)
            reference = kornia.color.rgb_to_yuv(reference)
        elif "XYZ" == color_space:
            image = kornia.color.rgb_to_xyz(image)
            reference = kornia.color.rgb_to_xyz(reference)

        image_mean, image_std = self.compute_mean_std(image)
        reference_mean, reference_std = self.compute_mean_std(reference)
        out = ((image - image_mean) / (image_std + 1e-6)) * (reference_std + 1e-6) + reference_mean
        out = factor * out + (1 - factor) * image

        if "LAB" == color_space:
            out = kornia.color.lab_to_rgb(out)
        elif "YCbCr" == color_space:
            out = kornia.color.ycbcr_to_rgb(out)
        elif "LUV" == color_space:
            out = kornia.color.luv_to_rgb(out)
        elif "YUV" == color_space:
            out = kornia.color.yuv_to_rgb(out)
        elif "XYZ" == color_space:
            out = kornia.color.xyz_to_rgb(out)

        out = out.permute([0, 2, 3, 1]).clamp(0, 1).to(comfy.model_management.intermediate_device())

        return (out,)

    def compute_mean_std(self, image):
        mean = torch.mean(image, dim=(2, 3), keepdim=True)
        std = torch.std(image, dim=(2, 3), keepdim=True)
        return mean, std

```
