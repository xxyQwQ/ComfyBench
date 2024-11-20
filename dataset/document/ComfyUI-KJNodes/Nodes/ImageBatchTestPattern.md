---
tags:
- Batch
- Image
- ImageBatch
---

# Image Batch Test Pattern
## Documentation
- Class name: `ImageBatchTestPattern`
- Category: `KJNodes/text`
- Output node: `False`

The ImageBatchTestPattern node is designed to generate a batch of images, each containing a unique sequential number displayed in a randomly colored font. This functionality is useful for creating test patterns in image batches, where each image is customized with specific text, font, and size settings, and positioned according to given coordinates.
## Input types
### Required
- **`batch_size`**
    - Specifies the number of images to generate in the batch, affecting the output batch size.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`start_from`**
    - The starting sequential number for the first image in the batch, setting the initial point for the sequence.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`text_x`**
    - The x-coordinate for the position of the text within the images, determining where the text appears horizontally.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`text_y`**
    - The y-coordinate for the position of the text within the images, determining where the text appears vertically.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`width`**
    - The width of the generated images, defining the horizontal dimension of each image in the batch.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - The height of the generated images, defining the vertical dimension of each image in the batch.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`font`**
    - The name of the font to use for the text in the images, influencing the visual style of the numbers.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`font_size`**
    - Determines the size of the font used for the text, directly impacting the appearance of the numbers in the images.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - A tensor representing the batch of generated images, each normalized and containing a unique sequential number in a randomly colored font.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ImageBatchTestPattern:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "batch_size": ("INT", {"default": 1,"min": 1, "max": 255, "step": 1}),
            "start_from": ("INT", {"default": 0,"min": 0, "max": 255, "step": 1}),
            "text_x": ("INT", {"default": 256,"min": 0, "max": 4096, "step": 1}),
            "text_y": ("INT", {"default": 256,"min": 0, "max": 4096, "step": 1}),
            "width": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
            "height": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
            "font": (folder_paths.get_filename_list("kjnodes_fonts"), ),
            "font_size": ("INT", {"default": 255,"min": 8, "max": 4096, "step": 1}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generatetestpattern"
    CATEGORY = "KJNodes/text"

    def generatetestpattern(self, batch_size, font, font_size, start_from, width, height, text_x, text_y):
        out = []
        # Generate the sequential numbers for each image
        numbers = np.arange(start_from, start_from + batch_size)
        font_path = folder_paths.get_full_path("kjnodes_fonts", font)

        for number in numbers:
            # Create a black image with the number as a random color text
            image = Image.new("RGB", (width, height), color='black')
            draw = ImageDraw.Draw(image)
            
            # Generate a random color for the text
            font_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            
            font = ImageFont.truetype(font_path, font_size)
            
            # Get the size of the text and position it in the center
            text = str(number)
           
            try:
                draw.text((text_x, text_y), text, font=font, fill=font_color, features=['-liga'])
            except:
                draw.text((text_x, text_y), text, font=font, fill=font_color,)
            
            # Convert the image to a numpy array and normalize the pixel values
            image_np = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np).unsqueeze(0)
            out.append(image_tensor)
        out_tensor = torch.cat(out, dim=0)
  
        return (out_tensor,)

```
