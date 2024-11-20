---
tags:
- Scheduling
- VisualEffects
---

# Parallax Motion Camera Scheduler
## Documentation
- Class name: `SaltLayerScheduler`
- Category: `SALT/AudioViz/Scheduling/Parallax Motion`
- Output node: `False`

The SaltLayerScheduler node is designed for scheduling parallax motion effects in audio-visual presentations, enabling dynamic camera movements to enhance the visual storytelling experience.
## Input types
### Required
- **`frame_count`**
    - Specifies the total number of frames for the parallax motion effect, determining the duration of the animation.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`zoom_speed`**
    - Controls the speed of zooming in or out, affecting the dynamic feel of the parallax motion.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`pan_speed`**
    - Sets the speed of panning, influencing the smoothness and pace of horizontal movement.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`pan_directions`**
    - Defines the directions for panning, allowing for customization of the motion path.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`direction_change_frames`**
    - Determines the frames at which the direction of panning changes, enabling complex motion patterns.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`tremor_scale`**
    - Adjusts the scale of tremor effects, adding a subtle dynamic to the visual presentation.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`tremor_octaves`**
    - Modifies the complexity of the tremor effect through octaves, enhancing the visual texture.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`tremor_persistence`**
    - Influences the persistence of tremor effects, affecting their visual impact over time.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`tremor_lacunarity`**
    - Controls the gap between successive tremor effects, shaping the overall motion aesthetic.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`direction_curve`**
    - Specifies the easing function for direction changes, smoothing transitions between movements.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `list`
- **`start_x`**
    - Sets the initial X position for the motion, establishing the starting point of the animation.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`start_y`**
    - Determines the initial Y position, marking the beginning of the vertical motion.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`zoom_mode`**
    - Selects the zooming mode, offering various options for zoom dynamics.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `list`
- **`layer_offsets`**
    - Defines offsets for different layers, creating depth in the parallax effect.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`schedule_list`**
    - Comfy dtype: `LIST`
    - Outputs a list representing the scheduled parallax motion effects, detailing the sequence of camera movements and adjustments.
    - Python dtype: `list`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltLayerScheduler:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frame_count": ("INT", {"default": 60, "min": 1, "max": 4096}),
                "zoom_speed": ("FLOAT", {"default": 0.01, "min": 0.001, "max": 1.0, "step": 0.001}),
                "pan_speed": ("FLOAT", {"default": 0.5, "min": 0.001, "max": 5.0, "step": 0.001}),
                "pan_directions": ("STRING", {"default": "90,180,270"}),
                "direction_change_frames": ("STRING", {"default": "10,20,40"}),
                "tremor_scale": ("FLOAT", {"default": 64, "min": 0.01, "max": 1024.0, "step": 0.01}),
                "tremor_octaves": ("INT", {"default": 1, "min": 1, "max": 10}),
                "tremor_persistence": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0, "step": 0.01}),
                "tremor_lacunarity": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 3.0, "step": 0.01}),
                "direction_curve": (list(easing_functions.keys()), ),
                "start_x": ("FLOAT", {"default": 0, "min": -nodes.MAX_RESOLUTION, "max": nodes.MAX_RESOLUTION}),
                "start_y": ("FLOAT", {"default": 0}),
                "zoom_mode": (["zoom-in", "zoom-out", "zoom-in-out"], ),
                "layer_offsets": ("STRING", {"default": "1,0.8,0.6"}),
            }
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("schedule_list",)
    FUNCTION = "execute"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Scheduling/Parallax Motion"

    def __init__(self):
        self.scheduler = None

    def execute(self, **kwargs):

        self.process_kwargs(**kwargs)

        if not self.scheduler:
            raise Exception("Camera Scheduler was not initialized. Perhaps your settings are bugged?")
        
        mode = kwargs.get("zoom_mode", "zoom-in")
        layer_offsets = kwargs.get("layer_offsets", "1")

        animation_data = self.scheduler.animate(mode, layer_offsets)

        return (animation_data, )

    def process_kwargs(self, **kwargs):
        frame_count = kwargs.get("frame_count", 60)
        zoom_speed = kwargs.get("zoom_speed", 0.01)
        pan_speed = kwargs.get("pan_speed", 0.5)
        pan_directions = list(map(float, kwargs.get("pan_directions", "90,180,270").split(",")))
        direction_change_frames = list(map(int, kwargs.get("direction_change_frames", "10,20,40").split(",")))
        tremor_params = {
            "scale": kwargs.get("tremor_scale", 0.1),
            "octaves": kwargs.get("tremor_octaves", 1),
            "persistence": kwargs.get("tremor_persistence", 0.5),
            "lacunarity": kwargs.get("tremor_lacunarity", 2.0),
        }
        direction_curve = kwargs.get("direction_curve", "linear")
        start_x = kwargs.get("start_x", 0)
        start_y = kwargs.get("start_y", 0)

        self.scheduler = OrganicPerlinCameraScheduler(
            frame_count=frame_count,
            zoom_speed=zoom_speed,
            pan_speed=pan_speed,
            pan_directions=pan_directions,
            direction_change_frames=direction_change_frames,
            tremor_params=tremor_params,
            direction_curve=direction_curve,
            start_x=start_x,
            start_y=start_y,
        )

```
