---
tags:
- Image
- TextOnImage
---

# LayerUtility: TextImage
## Documentation
- Class name: `LayerUtility: TextImage`
- Category: `😺dzNodes/LayerUtility`
- Output node: `False`

This node is designed to generate images from text by applying specific font styles, sizes, and colors. It allows for the customization of text appearance on images, including the layout and spacing, to create visually appealing graphics or text overlays on images. The node leverages image processing techniques to convert text into a graphical representation, suitable for a variety of applications such as dynamic image generation, graphic design, and content creation.
## Input types
### Required
- **`text`**
    - The text to be converted into an image. This parameter is crucial as it defines the content that will be visually represented in the generated image.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`font_file`**
    - Specifies the font file to use for the text rendering. This affects the style and appearance of the text in the image, allowing for a wide range of typographic designs.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`spacing`**
    - Controls the spacing between characters in the text. Adjusting this parameter can enhance readability or achieve a desired stylistic effect.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`leading`**
    - Determines the leading (line spacing) of the text, which is the distance between the baselines of successive lines of type. This parameter helps in adjusting the layout of text blocks for better visual composition.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`horizontal_border`**
    - The horizontal border padding for the text within the image. This parameter ensures that the text does not touch the edges of the image, providing a cleaner look.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`vertical_border`**
    - The vertical border padding for the text within the image. Similar to the horizontal border, it prevents the text from touching the top and bottom edges of the image.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`scale`**
    - Affects the overall size of the text in the image. Scaling the text up or down can help it fit better within the image or achieve a specific visual impact.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`variation_range`**
    - Defines the range of variation for random adjustments to the text appearance, such as size or position, to add uniqueness or dynamism to the generated image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`variation_seed`**
    - The seed for generating random variations in text appearance, ensuring reproducibility of the text styling when needed.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`layout`**
    - Specifies the layout orientation of the text within the image, such as horizontal or vertical alignment, influencing the overall composition and readability.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`width`**
    - The width of the image canvas on which the text is rendered. This parameter defines the horizontal boundary for the text layout and appearance.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - The height of the image canvas on which the text is rendered. This parameter defines the vertical boundary for the text layout and appearance.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`text_color`**
    - The color of the text. This parameter allows for customization of the text's visual impact against the image background.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`background_color`**
    - The background color of the image. It sets the overall tone of the image and can affect the contrast and visibility of the text.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
- **`size_as`**
    - An optional parameter that allows the image size to be defined relative to another element or parameter, providing flexibility in image scaling and layout.
    - Comfy dtype: `*`
    - Python dtype: `any`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The generated image with the text rendered according to the specified parameters. This output is the primary result of the node's processing, showcasing the text as a visual element within an image.
    - Python dtype: `torch.Tensor`
- **`mask`**
    - Comfy dtype: `MASK`
    - A mask that represents the area of the text within the generated image. This can be useful for further image processing tasks, such as compositing or applying effects specifically to the text area.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class TextImage:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        layout_list = ['horizontal', 'vertical']
        random_seed = int(time.time())
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "Text"}),
                "font_file": (FONT_LIST,),
                "spacing": ("INT", {"default": 0, "min": -9999, "max": 9999, "step": 1}),
                "leading": ("INT", {"default": 0, "min": -9999, "max": 9999, "step": 1}),
                "horizontal_border": ("FLOAT", {"default": 5, "min": -100, "max": 100, "step": 0.1}), # 左右距离百分比，横排为距左侧，竖排为距右侧
                "vertical_border": ("FLOAT", {"default": 5, "min": -100, "max": 100, "step": 0.1}),  # 上距离百分比
                "scale": ("FLOAT", {"default": 80, "min": 0.1, "max": 999, "step": 0.01}),  # 整体大小与画面长宽比，横排与宽比，竖排与高比
                "variation_range": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}), # 随机大小和位置范围
                "variation_seed": ("INT", {"default": random_seed, "min": 0, "max": 999999999999, "step": 1}),  # 随机种子
                "layout": (layout_list,),  # 横排or竖排
                "width": ("INT", {"default": 512, "min": 4, "max": 999999, "step": 1}),
                "height": ("INT", {"default": 512, "min": 4, "max": 999999, "step": 1}),
                "text_color": ("STRING", {"default": "#FFA000"}),  # 文字颜色
                "background_color": ("STRING", {"default": "#FFFFFF"}),  # 背景颜色
            },
            "optional": {
                "size_as": (any, {}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = 'text_image'
    CATEGORY = '😺dzNodes/LayerUtility'

    def text_image(self, text, font_file, spacing, leading, horizontal_border, vertical_border, scale,
                  variation_range, variation_seed, layout, width, height, text_color, background_color,
                  size_as=None
                  ):

        # spacing -= 20
        # leading += 20
        # scale *= 0.7
        if size_as is not None:
            width, height = tensor2pil(size_as).size
        text_table = []
        max_char_in_line = 0
        total_char = 0
        spacing = int(spacing * scale / 100)
        leading = int(leading * scale / 100)
        lines = []
        text_lines = text.split("\n")
        for l in text_lines:
            if len(l) > 0:
                lines.append(l)
                total_char += len(l)
                if len(l) > max_char_in_line:
                    max_char_in_line = len(l)
            else:
                lines.append(" ")
        if layout == 'vertical':
            char_horizontal_size = width // len(lines)
            char_vertical_size = height // max_char_in_line
            char_size = min(char_horizontal_size, char_vertical_size)
            if char_size < 1:
                char_size = 1
            start_x = width - int(width * horizontal_border/100) - char_size
        else:
            char_horizontal_size = width // max_char_in_line
            char_vertical_size = height // len(lines)
            char_size = min(char_horizontal_size, char_vertical_size)
            if char_size < 1:
                char_size = 1
            start_x = int(width * horizontal_border/100)
        start_y = int(height * vertical_border/100)

        # calculate every char position and size to a table list
        for i in range(len(lines)):
            _x = start_x
            _y = start_y
            line_table = []
            line_random = random_numbers(total=len(lines[i]),
                                         random_range=int(char_size * variation_range / 25),
                                         seed=variation_seed, sum_of_numbers=0)
            for j in range(0, len(lines[i])):
                offset = int((char_size + line_random[j]) * variation_range / 250)
                offset = int(offset * scale / 100)
                font_size = char_size + line_random[j]
                font_size = int(font_size * scale / 100)
                if font_size < 4:
                    font_size = 4
                axis_x = _x + offset // 3 if random.random() > 0.5 else _x - offset // 3
                axis_y = _y + offset // 3 if random.random() > 0.5 else _y - offset // 3
                char_dict = {'char':lines[i][j],
                             'axis':(axis_x, axis_y),
                             'size':font_size}
                line_table.append(char_dict)
                if layout == 'vertical':
                    _y += char_size + line_random[j] + spacing
                else:
                    _x += char_size + line_random[j] + spacing
            if layout == 'vertical':
                start_x -= leading * (i+1) + char_size
            else:
                start_y += leading * (i+1) + char_size
            text_table.append(line_table)

        # draw char
        _mask = Image.new('RGB', size=(width, height), color='black')
        draw = ImageDraw.Draw(_mask)
        for l in range(len(lines)):
            for c in range(len(lines[l])):
                font_path = FONT_DICT.get(font_file)
                font_size = text_table[l][c].get('size')
                font = ImageFont.truetype(font_path, font_size)
                draw.text(text_table[l][c].get('axis'), text_table[l][c].get('char'), font=font, fill='white')
        _canvas = Image.new('RGB', size=(width, height), color=background_color)
        _color = Image.new('RGB', size=(width, height), color=text_color)
        _canvas.paste(_color, mask=_mask.convert('L'))
        _canvas = RGB2RGBA(_canvas, _mask)
        log(f"{NODE_NAME} Processed.", message_type='finish')
        return (pil2tensor(_canvas), image2mask(_mask),)

```
