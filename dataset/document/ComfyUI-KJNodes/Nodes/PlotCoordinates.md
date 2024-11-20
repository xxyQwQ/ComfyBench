# Plot Coordinates
## Documentation
- Class name: `PlotCoordinates`
- Category: `KJNodes/experimental`
- Output node: `False`

The PlotCoordinates node is designed to process and manipulate coordinate data for visualization purposes. It adjusts bounding box sizes based on a size multiplier, ensures coordinates fit within specified frame boundaries, and generates a tensor representation of the plotted coordinates. This node is essential for visualizing spatial data, such as object locations in an image, by dynamically adjusting and plotting the coordinates according to given parameters.
## Input types
### Required
- **`coordinates`**
    - A JSON string or list of dictionaries representing the coordinates to be plotted. Each dictionary should contain 'x' and 'y' keys with their respective values. This input is crucial for determining the spatial locations to be visualized.
    - Comfy dtype: `STRING`
    - Python dtype: `Union[str, List[Dict[str, float]]]`
- **`text`**
    - A text prompt or description associated with the coordinates. This parameter can be used for adding contextual information to the visualization.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`width`**
    - The width of the frame in which the coordinates are to be plotted. This parameter defines the horizontal boundary for the visualization.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - The height of the frame in which the coordinates are to be plotted. This parameter defines the vertical boundary for the visualization.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`bbox_width`**
    - The base width of the bounding boxes before adjustment with the size multiplier. It sets the initial horizontal size for each plotted coordinate.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`bbox_height`**
    - The base height of the bounding boxes before adjustment with the size multiplier. It sets the initial vertical size for each plotted coordinate.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`size_multiplier`**
    - A list of multipliers to adjust the size of each bounding box based on the index. It allows for dynamic resizing of the bounding boxes for each coordinate, enhancing the visualization's adaptability.
    - Comfy dtype: `FLOAT`
    - Python dtype: `List[float]`
## Output types
- **`images`**
    - Comfy dtype: `IMAGE`
    - A sequence of images representing the plotted coordinates within the specified frame. This output is essential for visualizing the adjusted and fitted coordinates.
    - Python dtype: `torch.Tensor`
- **`width`**
    - Comfy dtype: `INT`
    - The width of the output image frame.
    - Python dtype: `int`
- **`height`**
    - Comfy dtype: `INT`
    - The height of the output image frame.
    - Python dtype: `int`
- **`bbox_width`**
    - Comfy dtype: `INT`
    - The width of the bounding boxes in the output image.
    - Python dtype: `int`
- **`bbox_height`**
    - Comfy dtype: `INT`
    - The height of the bounding boxes in the output image.
    - Python dtype: `int`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class PlotCoordinates:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                              "coordinates": ("STRING", {"forceInput": True}),
                              "text": ("STRING", {"default": 'title', "multiline": False}),
                              "width": ("INT", {"default": 512, "min": 8, "max": 4096, "step": 8}),
                              "height": ("INT", {"default": 512, "min": 8, "max": 4096, "step": 8}),
                              "bbox_width": ("INT", {"default": 128, "min": 8, "max": 4096, "step": 8}),
                              "bbox_height": ("INT", {"default": 128, "min": 8, "max": 4096, "step": 8}),
                            },
                "optional": {"size_multiplier": ("FLOAT", {"default": [1.0], "forceInput": True})},
                }
    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "INT",)
    RETURN_NAMES = ("images", "width", "height", "bbox_width", "bbox_height",)
    FUNCTION = "append"
    CATEGORY = "KJNodes/experimental"
    DESCRIPTION = """
Plots coordinates to sequence of images using Matplotlib.  

"""

    def append(self, coordinates, text, width, height, bbox_width, bbox_height, size_multiplier=[1.0]):
        coordinates = json.loads(coordinates.replace("'", '"'))
        coordinates = [(coord['x'], coord['y']) for coord in coordinates]
        batch_size = len(coordinates)    
        if len(size_multiplier) != batch_size:
            size_multiplier = size_multiplier * (batch_size // len(size_multiplier)) + size_multiplier[:batch_size % len(size_multiplier)]

        plot_image_tensor = plot_coordinates_to_tensor(coordinates, height, width, bbox_height, bbox_width, size_multiplier, text)
        
        return (plot_image_tensor, width, height, bbox_width, bbox_height)

```
