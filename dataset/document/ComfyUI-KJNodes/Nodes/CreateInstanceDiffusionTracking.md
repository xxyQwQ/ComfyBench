# CreateInstanceDiffusionTracking
## Documentation
- Class name: `CreateInstanceDiffusionTracking`
- Category: `KJNodes/InstanceDiffusion`
- Output node: `False`

This node is designed for creating tracking data to be utilized with InstanceDiffusion, a process that involves generating and managing instance-specific diffusion prompts and tracking information. It facilitates the structured organization of diffusion prompts based on class identifiers and names, enabling targeted image generation and manipulation.
## Input types
### Required
- **`coordinates`**
    - Specifies the coordinates for tracking, essential for defining the spatial location and boundaries of the instance to be tracked.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`width`**
    - Defines the width of the tracking area, setting the horizontal scale for instance tracking.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - Sets the height of the tracking area, determining the vertical scale for instance tracking.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`bbox_width`**
    - Specifies the width of the bounding box for the instance, crucial for delineating the instance's horizontal boundaries.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`bbox_height`**
    - Defines the height of the bounding box for the instance, essential for outlining the instance's vertical boundaries.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`class_name`**
    - The name of the class for the instance being tracked, used for categorization and identification in the diffusion process.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`class_id`**
    - A unique identifier for the class of the instance, facilitating differentiation and specific targeting in the diffusion process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`prompt`**
    - The diffusion prompt associated with the instance, guiding the generation process with specific textual cues.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
- **`size_multiplier`**
    - A multiplier for adjusting the size of the tracking area, allowing for flexible scaling based on specific needs.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`fit_in_frame`**
    - A boolean indicating whether the instance should be adjusted to fit within the frame, ensuring visibility and proper placement.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`tracking`**
    - Comfy dtype: `TRACKING`
    - The generated tracking data, structured for use with InstanceDiffusion, containing detailed instance-specific information.
    - Python dtype: `dict`
- **`prompt`**
    - Comfy dtype: `STRING`
    - A composite prompt string, derived from the class-specific prompts provided, for use in the diffusion process.
    - Python dtype: `str`
- **`width`**
    - Comfy dtype: `INT`
    - The width of the tracking area, as defined by the input parameters.
    - Python dtype: `int`
- **`height`**
    - Comfy dtype: `INT`
    - The height of the tracking area, as determined by the input parameters.
    - Python dtype: `int`
- **`bbox_width`**
    - Comfy dtype: `INT`
    - The width of the bounding box for the instance, as specified in the inputs.
    - Python dtype: `int`
- **`bbox_height`**
    - Comfy dtype: `INT`
    - The height of the bounding box for the instance, as outlined in the inputs.
    - Python dtype: `int`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class CreateInstanceDiffusionTracking:
    
    RETURN_TYPES = ("TRACKING", "STRING", "INT", "INT", "INT", "INT",)
    RETURN_NAMES = ("tracking", "prompt", "width", "height", "bbox_width", "bbox_height",)
    FUNCTION = "tracking"
    CATEGORY = "KJNodes/InstanceDiffusion"
    DESCRIPTION = """
Creates tracking data to be used with InstanceDiffusion:  
https://github.com/logtd/ComfyUI-InstanceDiffusion  
  
InstanceDiffusion prompt format:  
"class_id.class_name": "prompt",  
for example:  
"1.head": "((head))",  
"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "coordinates": ("STRING", {"forceInput": True}),
                "width": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
                "height": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
                "bbox_width": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
                "bbox_height": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
                "class_name": ("STRING", {"default": "class_name"}),
                "class_id": ("INT", {"default": 0,"min": 0, "max": 255, "step": 1}),
                "prompt": ("STRING", {"default": "prompt", "multiline": True}),
        },
        "optional": {
            "size_multiplier": ("FLOAT", {"default": [1.0], "forceInput": True}),
            "fit_in_frame": ("BOOLEAN", {"default": True}),
        }
    } 

    def tracking(self, coordinates, class_name, class_id, width, height, bbox_width, bbox_height, prompt, size_multiplier=[1.0], fit_in_frame=True):
        # Define the number of images in the batch
        coordinates = coordinates.replace("'", '"')
        coordinates = json.loads(coordinates)

        tracked = {}
        tracked[class_name] = {}
        batch_size = len(coordinates)
        # Initialize a list to hold the coordinates for the current ID
        id_coordinates = []
        if len(size_multiplier) != batch_size:
                size_multiplier = size_multiplier * (batch_size // len(size_multiplier)) + size_multiplier[:batch_size % len(size_multiplier)]
        for i, coord in enumerate(coordinates):
            x = coord['x']
            y = coord['y']
            adjusted_bbox_width = bbox_width * size_multiplier[i]
            adjusted_bbox_height = bbox_height * size_multiplier[i]
            # Calculate the top left and bottom right coordinates
            top_left_x = x - adjusted_bbox_width // 2
            top_left_y = y - adjusted_bbox_height // 2
            bottom_right_x = x + adjusted_bbox_width // 2
            bottom_right_y = y + adjusted_bbox_height // 2

            if fit_in_frame:
                # Clip the coordinates to the frame boundaries
                top_left_x = max(0, top_left_x)
                top_left_y = max(0, top_left_y)
                bottom_right_x = min(width, bottom_right_x)
                bottom_right_y = min(height, bottom_right_y)
                # Ensure width and height are positive
                adjusted_bbox_width = max(1, bottom_right_x - top_left_x)
                adjusted_bbox_height = max(1, bottom_right_y - top_left_y)

                # Update the coordinates with the new width and height
                bottom_right_x = top_left_x + adjusted_bbox_width
                bottom_right_y = top_left_y + adjusted_bbox_height

            # Append the top left and bottom right coordinates to the list for the current ID
            id_coordinates.append([top_left_x, top_left_y, bottom_right_x, bottom_right_y, width, height])
        
        class_id = int(class_id)
        # Assign the list of coordinates to the specified ID within the class_id dictionary
        tracked[class_name][class_id] = id_coordinates

        prompt_string = ""
        for class_name, class_data in tracked.items():
            for class_id in class_data.keys():
                class_id_str = str(class_id)
                # Use the incoming prompt for each class name and ID
                prompt_string += f'"{class_id_str}.{class_name}": "({prompt})",\n'

        # Remove the last comma and newline
        prompt_string = prompt_string.rstrip(",\n")

        return (tracked, prompt_string, width, height, bbox_width, bbox_height)

```
