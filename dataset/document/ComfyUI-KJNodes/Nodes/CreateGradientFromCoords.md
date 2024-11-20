# Create Gradient From Coords
## Documentation
- Class name: `CreateGradientFromCoords`
- Category: `KJNodes/image`
- Output node: `False`

This node is designed to generate a gradient image based on specified start and end coordinates, colors, and a multiplier. It creates a visual transition between two colors across the image canvas, allowing for the creation of dynamic and customizable gradient backgrounds or elements.
## Input types
### Required
- **`coordinates`**
    - A string representing the start and end coordinates for the gradient. These coordinates dictate the direction and span of the gradient across the image.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`frame_width`**
    - The width of the output image in pixels. It defines the horizontal dimension of the gradient image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`frame_height`**
    - The height of the output image in pixels. It defines the vertical dimension of the gradient image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`start_color`**
    - The color at the start of the gradient, specified as a string. This color marks the beginning of the gradient effect.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`end_color`**
    - The color at the end of the gradient, specified as a string. This color marks the completion of the gradient effect.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`multiplier`**
    - A float that adjusts the intensity of the gradient effect, allowing for finer control over the gradient's appearance.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The generated gradient image, returned as a tensor. This image visually represents the gradient created from the specified coordinates and colors.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class CreateGradientFromCoords:
    
    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image", )
    FUNCTION = "generate"
    CATEGORY = "KJNodes/image"
    DESCRIPTION = """
Creates a gradient image from coordinates.    
"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "coordinates": ("STRING", {"forceInput": True}),
                "frame_width": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
                "frame_height": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
                "start_color": ("STRING", {"default": 'white'}),
                "end_color": ("STRING", {"default": 'black'}),
                "multiplier": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01}),
        },
    } 
    
    def generate(self, coordinates, frame_width, frame_height, start_color, end_color, multiplier):
        # Parse the coordinates
        coordinates = json.loads(coordinates.replace("'", '"'))

        # Create an image
        image = Image.new("RGB", (frame_width, frame_height))
        draw = ImageDraw.Draw(image)

        # Extract start and end points for the gradient
        start_coord = coordinates[0]
        end_coord = coordinates[1]

        start_color = ImageColor.getrgb(start_color)
        end_color = ImageColor.getrgb(end_color)

        # Calculate the gradient direction (vector)
        gradient_direction = (end_coord['x'] - start_coord['x'], end_coord['y'] - start_coord['y'])
        gradient_length = (gradient_direction[0] ** 2 + gradient_direction[1] ** 2) ** 0.5

        # Iterate over each pixel in the image
        for y in range(frame_height):
            for x in range(frame_width):
                # Calculate the projection of the point on the gradient line
                point_vector = (x - start_coord['x'], y - start_coord['y'])
                projection = (point_vector[0] * gradient_direction[0] + point_vector[1] * gradient_direction[1]) / gradient_length
                projection = max(min(projection, gradient_length), 0)  # Clamp the projection value

                # Calculate the blend factor for the current pixel
                blend = projection * multiplier / gradient_length 

                # Determine the color of the current pixel
                color = (
                    int(start_color[0] + (end_color[0] - start_color[0]) * blend),
                    int(start_color[1] + (end_color[1] - start_color[1]) * blend),
                    int(start_color[2] + (end_color[2] - start_color[2]) * blend)
                )

                # Set the pixel color
                draw.point((x, y), fill=color)

        # Convert the PIL image to a tensor (assuming such a function exists in your context)
        image_tensor = pil2tensor(image)

        return (image_tensor,)

```
