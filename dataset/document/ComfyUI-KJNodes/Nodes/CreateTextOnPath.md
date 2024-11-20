# Create Text On Path
## Documentation
- Class name: `CreateTextOnPath`
- Category: `KJNodes/masking/generate`
- Output node: `False`

This node is designed to create text along a specified path, allowing for dynamic text placement and styling in graphical applications. It enables the precise alignment and orientation of text to follow complex curves or shapes, enhancing the visual appeal and readability of text in graphical interfaces or designs.
## Input types
### Required
- **`coordinates`**
    - Specifies the path along which the text will be created. This is crucial for determining the trajectory of the text alignment and placement.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`text`**
    - The text to be placed along the path. This input is crucial for defining the content that will be visually represented in the design.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`frame_width`**
    - Defines the width of the frame within which the text is to be placed. It affects the scaling and positioning of the text relative to the path.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`frame_height`**
    - Defines the height of the frame within which the text is to be placed. It affects the scaling and positioning of the text relative to the path.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`font`**
    - Specifies the font used for the text. This input is essential for customizing the appearance of the text to match design requirements.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`font_size`**
    - Determines the size of the font. This is key to ensuring the text is legible and aesthetically pleasing when placed along the path.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`alignment`**
    - Defines the alignment of the text relative to the path. This is important for controlling how the text is positioned and oriented along the path.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`text_color`**
    - Specifies the color of the text. This input allows for the customization of the text's visual appearance to fit the overall design.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
- **`size_multiplier`**
    - Affects the overall size of the text, allowing for dynamic scaling based on design needs.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The generated image with text placed along the specified path, adhering to the input style and content requirements.
    - Python dtype: `str`
- **`mask`**
    - Comfy dtype: `MASK`
    - A mask representing the area of the image where text has been placed, useful for further image processing steps.
    - Python dtype: `str`
- **`mask_inverted`**
    - Comfy dtype: `MASK`
    - An inverted mask of the text area, indicating where text has not been placed, which can be useful for additional processing or effects.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class CreateTextOnPath:
    
    RETURN_TYPES = ("IMAGE", "MASK", "MASK",)
    RETURN_NAMES = ("image", "mask", "mask_inverted",)
    FUNCTION = "createtextmask"
    CATEGORY = "KJNodes/masking/generate"
    DESCRIPTION = """
Creates a mask or batch of masks with the specified text.  
Locations are center locations.  
"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "coordinates": ("STRING", {"forceInput": True}),
                "text": ("STRING", {"default": 'text', "multiline": True}),
                "frame_width": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
                "frame_height": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
                "font": (folder_paths.get_filename_list("kjnodes_fonts"), ),
                "font_size": ("INT", {"default": 42}),
                 "alignment": (
                [   'left',
                    'center',
                    'right'
                ],
                {"default": 'center'}
                ),
                "text_color": ("STRING", {"default": 'white'}),
        },
        "optional": {
            "size_multiplier": ("FLOAT", {"default": [1.0], "forceInput": True}),
        }
    } 

    def createtextmask(self, coordinates, frame_width, frame_height, font, font_size, text, text_color, alignment, size_multiplier=[1.0]):
        coordinates = coordinates.replace("'", '"')
        coordinates = json.loads(coordinates)

        batch_size = len(coordinates)
        mask_list = []
        image_list = []
        color = text_color
        font_path = folder_paths.get_full_path("kjnodes_fonts", font)

        if len(size_multiplier) != batch_size:
            size_multiplier = size_multiplier * (batch_size // len(size_multiplier)) + size_multiplier[:batch_size % len(size_multiplier)]
        
        for i, coord in enumerate(coordinates):
            image = Image.new("RGB", (frame_width, frame_height), "black")
            draw = ImageDraw.Draw(image)
            lines = text.split('\n')  # Split the text into lines
            # Apply the size multiplier to the font size for this iteration
            current_font_size = int(font_size * size_multiplier[i])
            current_font = ImageFont.truetype(font_path, current_font_size)
            line_heights = [current_font.getbbox(line)[3] for line in lines]  # List of line heights
            total_text_height = sum(line_heights)  # Total height of text block

            # Calculate the starting Y position to center the block of text
            start_y = coord['y'] - total_text_height // 2
            for j, line in enumerate(lines):
                text_width, text_height = current_font.getbbox(line)[2], line_heights[j]
                if alignment == 'left':
                    location_x = coord['x']
                elif alignment == 'center':
                    location_x = int(coord['x'] - text_width // 2)
                elif alignment == 'right':
                    location_x = int(coord['x'] - text_width)
                
                location_y = int(start_y + sum(line_heights[:j]))
                text_position = (location_x, location_y)
                # Draw the text
                try:
                    draw.text(text_position, line, fill=color, font=current_font, features=['-liga'])
                except:
                    draw.text(text_position, line, fill=color, font=current_font)
            
            image = pil2tensor(image)
            non_black_pixels = (image > 0).any(dim=-1)
            mask = non_black_pixels.to(image.dtype)
            mask_list.append(mask)
            image_list.append(image)

        out_images = torch.cat(image_list, dim=0).cpu().float()
        out_masks = torch.cat(mask_list, dim=0)
        return (out_images, out_masks, 1.0 - out_masks,)

```
