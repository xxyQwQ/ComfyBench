---
tags:
- Color
---

# Colors Correct Node
## Documentation
- Class name: `ColorsCorrectNode`
- Category: `AlekPet Nodes/extras`
- Output node: `False`

The ColorsCorrectNode is designed for image color correction, allowing adjustments to brightness, contrast, saturation, gamma, and hue. It supports optional color tinting using a specified hex color value, providing a comprehensive suite of image enhancement tools.
## Input types
### Required
- **`image`**
    - The input image to be corrected. It serves as the base for all subsequent color and enhancement operations.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`brightness`**
    - Adjusts the brightness of the image. A higher value increases brightness.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`contrast`**
    - Adjusts the contrast of the image. A higher value increases contrast.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`saturation`**
    - Adjusts the saturation of the image, enhancing or muting the colors.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`gamma`**
    - Adjusts the gamma of the image, affecting the luminance of mid-tones.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`hue_degrees`**
    - Adjusts the hue of the image in degrees. This shifts all colors in the image.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`use_color`**
    - A boolean flag that determines whether to apply a color tint to the image based on the specified hex color.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
- **`hex_color`**
    - The hex color code used for tinting the image if use_color is True. It allows for custom color adjustments.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The corrected image after applying the specified adjustments.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ColorsCorrectNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "brightness": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.05},
                ),
                "contrast": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.05},
                ),
                "saturation": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.05},
                ),
                "gamma": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.05},
                ),
                "hue_degrees": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 360.0, "step": 0.01},
                ),
                "use_color": (
                    "BOOLEAN",
                    {"default": True},
                ),
            },
            "optional": {
                "hex_color": (
                    "STRING",
                    {"default": "#00FF33"},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "correct"
    CATEGORY = "AlekPet Nodes/extras"

    @staticmethod
    def hex_to_rgb(hex_color):
        return ImageColor.getcolor(hex_color, "RGB")

    @staticmethod
    def hex_to_hue(hex_color):
        rgb = ImageColor.getcolor(hex_color, "RGB")
        r, g, b = [x / 255.0 for x in rgb]
        mx = max(r, g, b)
        mn = min(r, g, b)
        df = mx - mn

        if mx == mn:
            h = 0
        elif mx == r:
            h = (60 * ((g - b) / df) + 360) % 360
        elif mx == g:
            h = (60 * ((b - r) / df) + 120) % 360
        elif mx == b:
            h = (60 * ((r - g) / df) + 240) % 360

        return h

    @staticmethod
    def adjust_brightness(image, factor):
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)

    @staticmethod
    def adjust_contrast(image, factor):
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)

    @staticmethod
    def adjust_saturation(image, factor):
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(factor)

    @staticmethod
    def adjust_gamma(image, gamma):
        inv_gamma = 1.0 / gamma
        lut = [pow(x / 255.0, inv_gamma) * 255 for x in range(256)]
        lut = np.array(lut * 3, dtype=np.uint8)
        return image.point(lut)

    @staticmethod
    def degrees_to_hue(degrees):
        degrees = degrees % 360
        hue = degrees / 360.0

        if hue > 0.5:
            hue -= 1.0
        if hue < -0.5:
            hue += 1.0

        return hue

    @staticmethod
    def adjust_hue(image, hue):
        if not (-0.5 <= hue <= 0.5):
            raise ValueError("hue value is not in [-0.5, 0.5].")

        image_array = np.array(image.convert("RGB"), dtype=np.uint8)

        hsv_image = Image.fromarray(image_array).convert("HSV")
        hsv_array = np.array(hsv_image)

        hsv_array[..., 0] = (hsv_array[..., 0].astype(int) + int(hue * 255)) % 256

        rgb_image = Image.fromarray(hsv_array, mode="HSV").convert("RGB")
        return rgb_image

    @staticmethod
    def tint_image(image, hex_color):
        return ImageOps.colorize(image.convert("L"), black="black", white=hex_color)

    def correct(
        self,
        image,
        use_color=True,
        hex_color="#00FF33",
        brightness=1.0,
        contrast=1.0,
        saturation=1.0,
        gamma=1.0,
        hue_degrees=0.0,
    ):
        i = 255.0 * image[0].cpu().numpy()
        image = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

        if use_color:
            image = ColorsCorrectNode.tint_image(image, hex_color)

        image = ColorsCorrectNode.adjust_brightness(image, brightness)
        image = ColorsCorrectNode.adjust_contrast(image, contrast)
        image = ColorsCorrectNode.adjust_saturation(image, saturation)
        image = ColorsCorrectNode.adjust_gamma(image, gamma)

        hue_norm = ColorsCorrectNode.degrees_to_hue(hue_degrees)
        image = ColorsCorrectNode.adjust_hue(image, hue_norm)

        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]

        return (image,)

```
