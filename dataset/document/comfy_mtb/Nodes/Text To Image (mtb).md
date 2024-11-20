---
tags:
- Image
- TextOnImage
---

# Text To Image (mtb)
## Documentation
- Class name: `Text To Image (mtb)`
- Category: `mtb/generate`
- Output node: `False`

The Text To Image node is designed to convert text into an image using specified fonts. It searches for font files within a specified directory and utilizes these fonts to render the given text into an image format. This functionality allows for dynamic text visualization with various font styles, supporting a range of font file formats.
## Input types
### Required
- **`text`**
    - The text string to be converted into an image. This parameter is crucial as it defines the content that will be visually represented in the resulting image.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`font`**
    - Specifies the font to be used for rendering the text into an image. This parameter allows for customization of the text's appearance, enhancing the visual appeal of the output.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`wrap`**
    - Determines whether the text should wrap to fit within the specified width of the image. This parameter is important for ensuring the text fits neatly within the image boundaries without being cut off.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`trim`**
    - Indicates whether whitespace around the text should be trimmed in the final image. This parameter helps in creating a cleaner, more focused visual representation of the text.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`line_height`**
    - Controls the spacing between lines of text when the text wraps. This parameter is essential for maintaining readability and aesthetic spacing in multi-line text images.
    - Comfy dtype: `FLOAT`
    - Python dtype: `int`
- **`font_size`**
    - The size of the font used to render the text. This parameter directly affects the readability and visual impact of the text in the image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`width`**
    - The desired width of the output image. This parameter defines the horizontal dimension of the image canvas.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - The desired height of the output image. This parameter defines the vertical dimension of the image canvas.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`color`**
    - The color of the text in the image. This parameter allows for customization of the text color, enabling contrast and visibility against the background.
    - Comfy dtype: `COLOR`
    - Python dtype: `str`
- **`background`**
    - The background color of the image. This parameter is crucial for setting the overall tone and contrast of the image, affecting the visibility of the text.
    - Comfy dtype: `COLOR`
    - Python dtype: `str`
- **`h_align`**
    - Horizontal alignment of the text within the image. This parameter controls the text's positioning along the horizontal axis, affecting the layout and balance of the image.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`v_align`**
    - Vertical alignment of the text within the image. This parameter controls the text's positioning along the vertical axis, affecting the layout and balance of the image.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`h_offset`**
    - Horizontal offset for the text position from its aligned location. This parameter allows for fine-tuning of the text's placement within the image, enabling precise layout adjustments.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`v_offset`**
    - Vertical offset for the text position from its aligned location. This parameter allows for fine-tuning of the text's placement within the image, enabling precise layout adjustments.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`h_coverage`**
    - Specifies the percentage of the image width that the text should cover. This parameter is important for controlling the text's spread across the image, affecting the visual density and readability.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output is an image with the specified text rendered using the chosen font and settings. This image represents the visual interpretation of the input text, customized according to the provided parameters.
    - Python dtype: `Image`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class MTB_TextToImage:
    """Utils to convert text to image using a font.

    The tool looks for any .ttf file in the Comfy folder hierarchy.
    """

    fonts = {}
    DESCRIPTION = """# Text to Image

This node look for any font files in comfy_dir/fonts.  
by default it fallsback to a default font.

![img](https://i.imgur.com/3GT92hy.gif)
"""

    def __init__(self):
        # - This is executed when the graph is executed,
        # - we could conditionaly reload fonts there
        pass

    @classmethod
    def CACHE_FONTS(cls):
        font_extensions = ["*.ttf", "*.otf", "*.woff", "*.woff2", "*.eot"]
        fonts = [font_path]

        for extension in font_extensions:
            try:
                if comfy_dir.exists():
                    fonts.extend(comfy_dir.glob(f"fonts/**/{extension}"))
                else:
                    log.warn(f"Directory {comfy_dir} does not exist.")
            except Exception as e:
                log.error(f"Error during font caching: {e}")

        for font in fonts:
            log.debug(f"Adding font {font}")
            MTB_TextToImage.fonts[font.stem] = font.as_posix()

    @classmethod
    def INPUT_TYPES(cls):
        if not cls.fonts:
            cls.CACHE_FONTS()
        else:
            log.debug(f"Using cached fonts (count: {len(cls.fonts)})")
        return {
            "required": {
                "text": (
                    "STRING",
                    {"default": "Hello world!"},
                ),
                "font": ((sorted(cls.fonts.keys())),),
                "wrap": ("BOOLEAN", {"default": True}),
                "trim": ("BOOLEAN", {"default": True}),
                "line_height": (
                    "FLOAT",
                    {"default": 1.0, "min": 0, "step": 0.1},
                ),
                "font_size": (
                    "INT",
                    {"default": 32, "min": 1, "max": 2500, "step": 1},
                ),
                "width": (
                    "INT",
                    {"default": 512, "min": 1, "max": 8096, "step": 1},
                ),
                "height": (
                    "INT",
                    {"default": 512, "min": 1, "max": 8096, "step": 1},
                ),
                "color": (
                    "COLOR",
                    {"default": "black"},
                ),
                "background": (
                    "COLOR",
                    {"default": "white"},
                ),
                "h_align": (("left", "center", "right"), {"default": "left"}),
                "v_align": (("top", "center", "bottom"), {"default": "top"}),
                "h_offset": (
                    "INT",
                    {"default": 0, "min": 0, "max": 8096, "step": 1},
                ),
                "v_offset": (
                    "INT",
                    {"default": 0, "min": 0, "max": 8096, "step": 1},
                ),
                "h_coverage": (
                    "INT",
                    {"default": 100, "min": 1, "max": 100, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "text_to_image"
    CATEGORY = "mtb/generate"

    def text_to_image(
        self,
        text: str,
        font,
        wrap,
        trim,
        line_height,
        font_size,
        width,
        height,
        color,
        background,
        h_align="left",
        v_align="top",
        h_offset=0,
        v_offset=0,
        h_coverage=100,
    ):
        import textwrap

        from PIL import Image, ImageDraw, ImageFont

        font_path = self.fonts[font]

        text = (
            text.encode("ascii", "ignore").decode().strip() if trim else text
        )
        # Handle word wrapping
        if wrap:
            wrap_width = (((width / 100) * h_coverage) / font_size) * 2
            lines = textwrap.wrap(text, width=wrap_width)
        else:
            lines = [text]
        font = ImageFont.truetype(font_path, size=font_size)
        log.debug(f"Lines: {lines}")
        img = Image.new("RGBA", (width, height), background)
        draw = ImageDraw.Draw(img)

        line_height_px = line_height * font_size

        # Vertical alignment
        if v_align == "top":
            y_text = v_offset
        elif v_align == "center":
            y_text = ((height - (line_height_px * len(lines))) // 2) + v_offset
        else:  # bottom
            y_text = (height - (line_height_px * len(lines))) - v_offset

        def get_width(line):
            if hasattr(font, "getsize"):
                return font.getsize(line)[0]
            else:
                return font.getlength(line)

        # Draw each line of text
        for line in lines:
            line_width = get_width(line)
            # Horizontal alignment
            if h_align == "left":
                x_text = h_offset
            elif h_align == "center":
                x_text = ((width - line_width) // 2) + h_offset
            else:  # right
                x_text = (width - line_width) - h_offset

            draw.text((x_text, y_text), line, fill=color, font=font)
            y_text += line_height_px

        return (pil2tensor(img),)

```
