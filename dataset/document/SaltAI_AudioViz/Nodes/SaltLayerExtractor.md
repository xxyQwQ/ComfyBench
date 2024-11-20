---
tags:
- Scheduling
- VisualEffects
---

# Parallax Motion Camera Scheduler Extractor
## Documentation
- Class name: `SaltLayerExtractor`
- Category: `SALT/AudioViz/Scheduling/Parallax Motion`
- Output node: `False`

The SaltLayerExtractor node is designed for extracting specific layer data from a sequence of animation layers based on a given index. It focuses on isolating zoom, x, and y values for a particular layer, facilitating the creation of detailed animation schedules for parallax motion effects.
## Input types
### Required
- **`float_layers`**
    - A list of animation layers, where each layer is represented as a sequence of frames with float values for zoom, x, and y positions. This input is crucial for determining the specific layer data to extract.
    - Comfy dtype: `LIST`
    - Python dtype: `List[List[float]]`
- **`layer_index`**
    - An integer specifying the index of the layer to extract from the list of animation layers. This determines which layer's data is processed and extracted.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`zoom_schedule_lsit`**
    - Comfy dtype: `LIST`
    - unknown
    - Python dtype: `unknown`
- **`x_schedule_list`**
    - Comfy dtype: `LIST`
    - A list of x position values extracted from the specified layer, representing the horizontal movement schedule for animation.
    - Python dtype: `List[float]`
- **`y_schedule_list`**
    - Comfy dtype: `LIST`
    - A list of y position values extracted from the specified layer, representing the vertical movement schedule for animation.
    - Python dtype: `List[float]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltLayerExtractor:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "float_layers": ("LIST", ),
                "layer_index": ("INT", {"default": 0, "min": 0})
            }
        }

    RETURN_TYPES = ("LIST", "LIST", "LIST")
    RETURN_NAMES = ("zoom_schedule_lsit", "x_schedule_list", "y_schedule_list")
    FUNCTION = "extract"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Scheduling/Parallax Motion"

    def extract(self, **kwargs):
        animation_data = kwargs.get("float_layers", [])
        layer_index = kwargs.get("layer_index", 0)

        if layer_index >= len(animation_data):
            raise ValueError("Layer index out of range.")

        selected_layer_data = animation_data[layer_index]
        zoom_values = [frame[0] for frame in selected_layer_data]
        x_values = [frame[1] for frame in selected_layer_data]
        y_values = [frame[2] for frame in selected_layer_data]

        logger.info("\033[1m\033[94mOPAC Schedule Curves:\033[0m")
        log_curve("Zoom Values", zoom_values)
        log_curve("X Values", x_values)
        log_curve("Y Values", y_values)

        return zoom_values, x_values, y_values

```
