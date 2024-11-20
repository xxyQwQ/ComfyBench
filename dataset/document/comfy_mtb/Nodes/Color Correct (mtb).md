---
tags:
- Color
---

# Color Correct (mtb)
## Documentation
- Class name: `Color Correct (mtb)`
- Category: `mtb/image processing`
- Output node: `False`

Provides a suite of color correction capabilities for images, allowing adjustments to gamma, contrast, exposure, offset, hue, saturation, and value to enhance or modify the image's appearance.
## Input types
### Required
- **`image`**
    - The input image to be color corrected. This is the primary data upon which all color correction operations are applied.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`clamp`**
    - A boolean flag indicating whether the color values should be clamped. Clamping ensures that the color values do not exceed the allowable range, maintaining image integrity.
    - Comfy dtype: `COMBO[BOOLEAN]`
    - Python dtype: `bool`
- **`gamma`**
    - Adjusts the gamma value of the image, affecting the luminance to either darken or lighten the image.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`contrast`**
    - Modifies the contrast level of the image, enhancing the difference between the light and dark areas.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`exposure`**
    - Alters the exposure level, simulating the effect of changing the amount of light the image is exposed to.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`offset`**
    - Applies an offset to the color values, shifting all colors by a fixed amount.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`hue`**
    - Changes the hue of the image, adjusting the overall color tone.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`saturation`**
    - Adjusts the saturation level, affecting the intensity of the colors in the image.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`value`**
    - Modifies the value (brightness) of the image, making it brighter or darker.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The color-corrected image, with adjustments applied as per the input parameters.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class MTB_ColorCorrect:
    """Various color correction methods"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "clamp": ([True, False], {"default": True}),
                "gamma": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01},
                ),
                "contrast": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01},
                ),
                "exposure": (
                    "FLOAT",
                    {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.01},
                ),
                "offset": (
                    "FLOAT",
                    {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.01},
                ),
                "hue": (
                    "FLOAT",
                    {"default": 0.0, "min": -0.5, "max": 0.5, "step": 0.01},
                ),
                "saturation": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01},
                ),
                "value": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "correct"
    CATEGORY = "mtb/image processing"

    @staticmethod
    def gamma_correction_tensor(image, gamma):
        gamma_inv = 1.0 / gamma
        return image.pow(gamma_inv)

    @staticmethod
    def contrast_adjustment_tensor(image, contrast):
        r, g, b = image.unbind(-1)
        
        # Using Adobe RGB luminance weights.
        luminance_image = 0.33 * r + 0.71 * g + 0.06 * b
        luminance_mean = torch.mean(luminance_image.unsqueeze(-1))

        # Blend original with mean luminance using contrast factor as blend ratio.
        contrasted = image * contrast + (1.0 - contrast) * luminance_mean        
        return torch.clamp(contrasted, 0.0, 1.0)

    @staticmethod
    def exposure_adjustment_tensor(image, exposure):
        return image * (2.0**exposure)

    @staticmethod
    def offset_adjustment_tensor(image, offset):
        return image + offset

    @staticmethod
    def hsv_adjustment(image: torch.Tensor, hue, saturation, value):
        images = tensor2pil(image)
        out = []
        for img in images:
            hsv_image = img.convert("HSV")

            h, s, v = hsv_image.split()

            h = h.point(lambda x: (x + hue * 255) % 256)
            s = s.point(lambda x: int(x * saturation))
            v = v.point(lambda x: int(x * value))

            hsv_image = Image.merge("HSV", (h, s, v))
            rgb_image = hsv_image.convert("RGB")
            out.append(rgb_image)
        return pil2tensor(out)

    @staticmethod
    def hsv_adjustment_tensor_not_working(
        image: torch.Tensor, hue, saturation, value
    ):
        """Abandonning for now"""
        image = image.squeeze(0).permute(2, 0, 1)

        max_val, _ = image.max(dim=0, keepdim=True)
        min_val, _ = image.min(dim=0, keepdim=True)
        delta = max_val - min_val

        hue_image = torch.zeros_like(max_val)
        mask = delta != 0.0

        r, g, b = image[0], image[1], image[2]
        hue_image[mask & (max_val == r)] = ((g - b) / delta)[
            mask & (max_val == r)
        ] % 6.0
        hue_image[mask & (max_val == g)] = ((b - r) / delta)[
            mask & (max_val == g)
        ] + 2.0
        hue_image[mask & (max_val == b)] = ((r - g) / delta)[
            mask & (max_val == b)
        ] + 4.0

        saturation_image = delta / (max_val + 1e-7)
        value_image = max_val

        hue_image = (hue_image + hue) % 1.0
        saturation_image = torch.where(
            mask, saturation * saturation_image, saturation_image
        )
        value_image = value * value_image

        c = value_image * saturation_image
        x = c * (1 - torch.abs((hue_image % 2) - 1))
        m = value_image - c

        prime_image = torch.zeros_like(image)
        prime_image[0] = torch.where(
            max_val == r, c, torch.where(max_val == g, x, prime_image[0])
        )
        prime_image[1] = torch.where(
            max_val == r, x, torch.where(max_val == g, c, prime_image[1])
        )
        prime_image[2] = torch.where(
            max_val == g, x, torch.where(max_val == b, c, prime_image[2])
        )

        rgb_image = prime_image + m

        rgb_image = rgb_image.permute(1, 2, 0).unsqueeze(0)

        return rgb_image

    def correct(
        self,
        image: torch.Tensor,
        clamp: bool,
        gamma: float = 1.0,
        contrast: float = 1.0,
        exposure: float = 0.0,
        offset: float = 0.0,
        hue: float = 0.0,
        saturation: float = 1.0,
        value: float = 1.0,
    ):
        # Apply color correction operations
        image = self.gamma_correction_tensor(image, gamma)
        image = self.contrast_adjustment_tensor(image, contrast)
        image = self.exposure_adjustment_tensor(image, exposure)
        image = self.offset_adjustment_tensor(image, offset)
        image = self.hsv_adjustment(image, hue, saturation, value)

        if clamp:
            image = torch.clamp(image, 0.0, 1.0)

        return (image,)

```
