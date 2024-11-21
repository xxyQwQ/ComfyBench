# DrawInstanceDiffusionTracking
## Documentation
- Class name: `DrawInstanceDiffusionTracking`
- Category: `KJNodes/InstanceDiffusion`
- Output node: `False`

This node is designed for visualizing tracking data on images by drawing bounding boxes and optionally text annotations. It leverages tracking information generated by the CreateInstanceDiffusionTracking node to overlay visual cues on images, enhancing the interpretability of tracking data.
## Input types
### Required
- **`image`**
    - The image on which tracking data will be visualized. It serves as the canvas for drawing bounding boxes and text annotations.
    - Comfy dtype: `IMAGE`
    - Python dtype: `IMAGE`
- **`tracking`**
    - Tracking data containing information about detected objects, used to draw bounding boxes and text annotations on the image.
    - Comfy dtype: `TRACKING`
    - Python dtype: `TRACKING`
- **`box_line_width`**
    - Specifies the thickness of the bounding boxes drawn around detected objects.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`draw_text`**
    - A boolean flag indicating whether to draw text annotations (class name and ID) above the bounding boxes.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`font`**
    - The font used for text annotations, allowing customization of the visual appearance of text.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`font_size`**
    - The size of the font used for text annotations, affecting the readability of text above bounding boxes.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The image with tracking data visualized through bounding boxes and optional text annotations.
    - Python dtype: `IMAGE`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class DrawInstanceDiffusionTracking:
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image", )
    FUNCTION = "draw"
    CATEGORY = "KJNodes/InstanceDiffusion"
    DESCRIPTION = """
Draws the tracking data from  
CreateInstanceDiffusionTracking -node.

"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "tracking": ("TRACKING", {"forceInput": True}),
                "box_line_width": ("INT", {"default": 2, "min": 1, "max": 10, "step": 1}),
                "draw_text": ("BOOLEAN", {"default": True}),
                "font": (folder_paths.get_filename_list("kjnodes_fonts"), ),
                "font_size": ("INT", {"default": 20}),
        },
    } 

    def draw(self, image, tracking, box_line_width, draw_text, font, font_size):
        import matplotlib.cm as cm

        modified_images = []
        
        colormap = cm.get_cmap('rainbow', len(tracking))
        if draw_text:
            font_path = folder_paths.get_full_path("kjnodes_fonts", font)
            font = ImageFont.truetype(font_path, font_size)

        # Iterate over each image in the batch
        for i in range(image.shape[0]):
            # Extract the current image and convert it to a PIL image
            current_image = image[i, :, :, :].permute(2, 0, 1)
            pil_image = transforms.ToPILImage()(current_image)
            
            draw = ImageDraw.Draw(pil_image)
            
            # Iterate over the bounding boxes for the current image
            for j, (class_name, class_data) in enumerate(tracking.items()):
                for class_id, bbox_list in class_data.items():
                    # Check if the current index is within the bounds of the bbox_list
                    if i < len(bbox_list):
                        bbox = bbox_list[i]
                        # Ensure bbox is a list or tuple before unpacking
                        if isinstance(bbox, (list, tuple)):
                            x1, y1, x2, y2, _, _ = bbox
                            # Convert coordinates to integers
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            # Generate a color from the rainbow colormap
                            color = tuple(int(255 * x) for x in colormap(j / len(tracking)))[:3]
                            # Draw the bounding box on the image with the generated color
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=box_line_width)
                            if draw_text:
                                # Draw the class name and ID as text above the box with the generated color
                                text = f"{class_id}.{class_name}"
                                # Calculate the width and height of the text
                                _, _, text_width, text_height = draw.textbbox((0, 0), text=text, font=font)
                                # Position the text above the top-left corner of the box
                                text_position = (x1, y1 - text_height)
                                draw.text(text_position, text, fill=color, font=font)
                        else:
                            print(f"Unexpected data type for bbox: {type(bbox)}")
            
            # Convert the drawn image back to a torch tensor and adjust back to (H, W, C)
            modified_image_tensor = transforms.ToTensor()(pil_image).permute(1, 2, 0)
            modified_images.append(modified_image_tensor)
        
        # Stack the modified images back into a batch
        image_tensor_batch = torch.stack(modified_images).cpu().float()
        
        return image_tensor_batch,

```