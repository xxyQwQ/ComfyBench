# Create Shape Image On Path
## Documentation
- Class name: `CreateShapeImageOnPath`
- Category: `KJNodes/image`
- Output node: `False`

This node is designed to generate images or a batch of images with specific shapes drawn on them. It allows for the creation of images with shapes such as circles, squares, and triangles at specified locations, sizes, and colors. The functionality extends to adjusting the shape's appearance through parameters like blur radius and intensity, enabling the creation of both sharp and soft-edged shapes on a customizable background.
## Input types
### Required
- **`shape`**
    - Specifies the shape to be drawn on the image. It can be a circle, square, or triangle, affecting the visual representation in the generated image.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`coordinates`**
    - Defines the center location for the shape on the image. It is crucial for positioning the shape accurately within the frame.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`frame_width`**
    - Sets the width of the frame for the image, determining the overall size of the generated image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`frame_height`**
    - Sets the height of the frame for the image, determining the overall size of the generated image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`shape_width`**
    - Determines the width of the shape to be drawn, allowing for size customization of the shape.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`shape_height`**
    - Determines the height of the shape to be drawn, allowing for size customization of the shape.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`shape_color`**
    - Defines the color of the shape, enabling visual customization.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`bg_color`**
    - Sets the background color of the image, allowing for further customization of the image's appearance.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`blur_radius`**
    - Applies a Gaussian blur to the shape, enabling the creation of soft edges. The radius affects the degree of blurriness.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`intensity`**
    - Adjusts the intensity of the shape's color, affecting its visual prominence on the image.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
### Optional
- **`size_multiplier`**
    - Modifies the size of the shape based on the multiplier, allowing for dynamic size adjustments in a batch of images.
    - Comfy dtype: `FLOAT`
    - Python dtype: `list[float]`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The generated image with the specified shape drawn on it.
    - Python dtype: `torch.Tensor`
- **`mask`**
    - Comfy dtype: `MASK`
    - A mask of the shape drawn on the image, useful for image processing and analysis.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class CreateShapeImageOnPath:
    
    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image","mask", )
    FUNCTION = "createshapemask"
    CATEGORY = "KJNodes/image"
    DESCRIPTION = """
Creates an image or batch of images with the specified shape.  
Locations are center locations.  
"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "shape": (
            [   'circle',
                'square',
                'triangle',
            ],
            {
            "default": 'circle'
             }),
                "coordinates": ("STRING", {"forceInput": True}),
                "frame_width": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
                "frame_height": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
                "shape_width": ("INT", {"default": 128,"min": 8, "max": 4096, "step": 1}),
                "shape_height": ("INT", {"default": 128,"min": 8, "max": 4096, "step": 1}),
                "shape_color": ("STRING", {"default": 'white'}),
                "bg_color": ("STRING", {"default": 'black'}),
                "blur_radius": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100, "step": 0.1}),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01}),
        },
        "optional": {
            "size_multiplier": ("FLOAT", {"default": [1.0], "forceInput": True}),
        }
    } 

    def createshapemask(self, coordinates, frame_width, frame_height, shape_width, shape_height, shape_color, 
                        bg_color, blur_radius, shape, intensity, size_multiplier=[1.0]):
        # Define the number of images in the batch
        coordinates = coordinates.replace("'", '"')
        coordinates = json.loads(coordinates)

        batch_size = len(coordinates)
        images_list = []
        masks_list = []

        if len(size_multiplier) != batch_size:
            size_multiplier = size_multiplier * (batch_size // len(size_multiplier)) + size_multiplier[:batch_size % len(size_multiplier)]
        for i, coord in enumerate(coordinates):
            image = Image.new("RGB", (frame_width, frame_height), bg_color)
            draw = ImageDraw.Draw(image)

            # Calculate the size for this frame and ensure it's not less than 0
            current_width = max(0, shape_width + i * size_multiplier[i])
            current_height = max(0, shape_height + i * size_multiplier[i])

            location_x = coord['x']
            location_y = coord['y']

            if shape == 'circle' or shape == 'square':
                # Define the bounding box for the shape
                left_up_point = (location_x - current_width // 2, location_y - current_height // 2)
                right_down_point = (location_x + current_width // 2, location_y + current_height // 2)
                two_points = [left_up_point, right_down_point]

                if shape == 'circle':
                    draw.ellipse(two_points, fill=shape_color)
                elif shape == 'square':
                    draw.rectangle(two_points, fill=shape_color)
                    
            elif shape == 'triangle':
                # Define the points for the triangle
                left_up_point = (location_x - current_width // 2, location_y + current_height // 2) # bottom left
                right_down_point = (location_x + current_width // 2, location_y + current_height // 2) # bottom right
                top_point = (location_x, location_y - current_height // 2) # top point
                draw.polygon([top_point, left_up_point, right_down_point], fill=shape_color)

            if blur_radius != 0:
                    image = image.filter(ImageFilter.GaussianBlur(blur_radius))

            image = pil2tensor(image)
            image = image * intensity
            mask = image[:, :, :, 0]
            masks_list.append(mask)
            images_list.append(image)
        out_images = torch.cat(images_list, dim=0).cpu().float()
        out_masks = torch.cat(masks_list, dim=0)
        return (out_images, out_masks)

```
