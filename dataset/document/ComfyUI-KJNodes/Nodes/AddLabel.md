# Add Label
## Documentation
- Class name: `AddLabel`
- Category: `KJNodes/text`
- Output node: `False`

The AddLabel node is designed to add textual labels to images, allowing for customization of text position, size, color, and font. It supports various directions for label placement, including overlaying text directly on the image or positioning it around the image's edges, enhancing the image's informational content or aesthetic appeal.
## Input types
### Required
- **`image`**
    - The input image to which the label will be added. This is the primary canvas for the label application.
    - Comfy dtype: `IMAGE`
    - Python dtype: `PIL.Image.Image`
- **`text_x`**
    - The x-coordinate for the starting point of the text on the image. It determines the horizontal position of the label.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`text_y`**
    - The y-coordinate for the starting point of the text on the image. It determines the vertical position of the label.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - The height of the label area in pixels. This can affect the overall size of the image if the label is not overlaid.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`font_size`**
    - The size of the font used for the label text, impacting the readability and visual impact of the label.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`font_color`**
    - The color of the font used for the label text, allowing for visual contrast or harmony with the image and label background.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`label_color`**
    - The background color of the label area, which can be used to highlight the text or integrate with the image's color scheme.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`font`**
    - The font used for the label text, selected from a predefined list of fonts. This choice influences the label's aesthetic and legibility.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`text`**
    - The text content of the label to be added to the image. This defines what the label will say.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`direction`**
    - The direction in which the label will be added relative to the image, including options like up, down, left, right, and overlay, affecting the label's placement and interaction with the image.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
### Optional
- **`caption`**
    - An optional caption to include with the label, providing additional context or information.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The image with the added label, reflecting all specified customizations.
    - Python dtype: `PIL.Image.Image`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class AddLabel:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image":("IMAGE",),  
            "text_x": ("INT", {"default": 10, "min": 0, "max": 4096, "step": 1}),
            "text_y": ("INT", {"default": 2, "min": 0, "max": 4096, "step": 1}),
            "height": ("INT", {"default": 48, "min": 0, "max": 4096, "step": 1}),
            "font_size": ("INT", {"default": 32, "min": 0, "max": 4096, "step": 1}),
            "font_color": ("STRING", {"default": "white"}),
            "label_color": ("STRING", {"default": "black"}),
            "font": (folder_paths.get_filename_list("kjnodes_fonts"), ),
            "text": ("STRING", {"default": "Text"}),
            "direction": (
            [   'up',
                'down',
                'left',
                'right',
                'overlay'
            ],
            {
            "default": 'up'
             }),
            },
            "optional":{
                "caption": ("STRING", {"default": "", "forceInput": True}),
            }
            }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "addlabel"
    CATEGORY = "KJNodes/text"
    DESCRIPTION = """
Creates a new with the given text, and concatenates it to  
either above or below the input image.  
Note that this changes the input image's height!  
Fonts are loaded from this folder:  
ComfyUI/custom_nodes/ComfyUI-KJNodes/fonts
"""
        
    def addlabel(self, image, text_x, text_y, text, height, font_size, font_color, label_color, font, direction, caption=""):
        batch_size = image.shape[0]
        width = image.shape[2]
        
        font_path = os.path.join(script_directory, "fonts", "TTNorms-Black.otf") if font == "TTNorms-Black.otf" else folder_paths.get_full_path("kjnodes_fonts", font)
        
        def process_image(input_image, caption_text):
            if direction == 'overlay':
                pil_image = Image.fromarray((input_image.cpu().numpy() * 255).astype(np.uint8))
            else:
                label_image = Image.new("RGB", (width, height), label_color)
                pil_image = label_image
                
            draw = ImageDraw.Draw(pil_image)
            font = ImageFont.truetype(font_path, font_size)
            
            words = caption_text.split()
            
            lines = []
            current_line = []
            current_line_width = 0
            for word in words:
                word_width = font.getbbox(word)[2]
                if current_line_width + word_width <= width - 2 * text_x:
                    current_line.append(word)
                    current_line_width += word_width + font.getbbox(" ")[2] # Add space width
                else:
                    lines.append(" ".join(current_line))
                    current_line = [word]
                    current_line_width = word_width
            
            if current_line:
                lines.append(" ".join(current_line))
            
            y_offset = text_y
            for line in lines:
                try:
                    draw.text((text_x, y_offset), line, font=font, fill=font_color, features=['-liga'])
                except:
                    draw.text((text_x, y_offset), line, font=font, fill=font_color)
                y_offset += font_size # Move to the next line
                
            processed_image = torch.from_numpy(np.array(pil_image).astype(np.float32) / 255.0).unsqueeze(0)
            return processed_image
        
        if caption == "":
            processed_images = [process_image(img, text) for img in image]
        else:
            assert len(caption) == batch_size, "Number of captions does not match number of images"
            processed_images = [process_image(img, cap) for img, cap in zip(image, caption)]
        processed_batch = torch.cat(processed_images, dim=0)

        # Combine images based on direction
        if direction == 'down':
            combined_images = torch.cat((image, processed_batch), dim=1)
        elif direction == 'up':
            combined_images = torch.cat((processed_batch, image), dim=1)
        elif direction == 'left':
            processed_batch = torch.rot90(processed_batch, 3, (2, 3)).permute(0, 3, 1, 2)
            combined_images = torch.cat((processed_batch, image), dim=2)
        elif direction == 'right':
            processed_batch = torch.rot90(processed_batch, 3, (2, 3)).permute(0, 3, 1, 2)
            combined_images = torch.cat((image, processed_batch), dim=2)
        else:
            combined_images = processed_batch
        
        return (combined_images,)

```
