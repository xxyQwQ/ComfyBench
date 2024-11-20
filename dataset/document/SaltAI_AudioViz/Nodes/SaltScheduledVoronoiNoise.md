---
tags:
- Noise
- NoisePatterns
---

# Scheduled Voronoi Noise Generator
## Documentation
- Class name: `SaltScheduledVoronoiNoise`
- Category: `SALT/AudioViz/Scheduling/Image`
- Output node: `False`

This node is designed to generate visual noise patterns based on the Voronoi diagram algorithm, with the ability to schedule various parameters over time. It allows for dynamic creation of complex, aesthetically pleasing visual textures that can be customized and animated according to specific schedules for scale, detail, randomness, and more.
## Input types
### Required
- **`batch_size`**
    - Specifies the number of noise patterns to generate in one execution. It affects the output's scale and can be used to create multiple instances of noise patterns simultaneously.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`width`**
    - Determines the width of the generated noise pattern. It directly influences the spatial resolution of the output texture.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - Sets the height of the generated noise pattern, affecting the spatial resolution of the output texture similarly to width.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`distance_metric`**
    - Chooses the method for calculating distances in the Voronoi diagram, affecting the visual characteristics of the noise pattern.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`x_schedule`**
    - A schedule for the x-axis positions over time, allowing for dynamic changes in the noise pattern's horizontal distribution.
    - Comfy dtype: `LIST`
    - Python dtype: `list`
- **`y_schedule`**
    - A schedule for the y-axis positions over time, enabling dynamic alterations in the vertical distribution of the noise pattern.
    - Comfy dtype: `LIST`
    - Python dtype: `list`
- **`scale_schedule`**
    - Controls the scale of the noise pattern over time, enabling zoom in/out effects.
    - Comfy dtype: `LIST`
    - Python dtype: `list`
- **`detail_schedule`**
    - Adjusts the level of detail in the noise pattern over time, affecting its complexity and texture.
    - Comfy dtype: `LIST`
    - Python dtype: `list`
- **`randomness_schedule`**
    - Modifies the randomness factor of the noise pattern over time, influencing its unpredictability and variation.
    - Comfy dtype: `LIST`
    - Python dtype: `list`
- **`seed_schedule`**
    - Schedules the seed values used for noise generation, allowing for controlled variations in the pattern.
    - Comfy dtype: `LIST`
    - Python dtype: `list`
- **`device`**
    - Specifies the computing device (CPU/GPU) on which the noise generation is executed, affecting performance.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`images`**
    - Comfy dtype: `IMAGE`
    - The generated noise patterns as a batch of images, ready for further processing or visualization.
    - Python dtype: `torch.Tensor`
- **`batch_size`**
    - Comfy dtype: `INT`
    - The number of generated noise patterns, confirming the input batch size was processed as requested.
    - Python dtype: `int`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class SaltScheduledVoronoiNoise:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch_size": ("INT", {"min": 1, "max": 4096}),
                "width": ("INT", {"default": 64, "min": 64, "max": MAX_RESOLUTION}),
                "height": ("INT", {"default": 64, "min": 64, "max": MAX_RESOLUTION}),
            }, 
            "optional": {
                "distance_metric": (["euclidean", "manhattan", "chebyshev", "minkowski"],),
                "x_schedule": ("LIST",),
                "y_schedule": ("LIST",),
                "scale_schedule": ("LIST",),
                "detail_schedule": ("LIST",),
                "randomness_schedule": ("LIST",),
                "seed_schedule": ("LIST", ),
                "device": (["cuda", "cpu"],),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "batch_size")

    FUNCTION = "generate"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Scheduling/Image"

    def generate(self, batch_size, width, height, distance_metric="euclidean", x_schedule=[0], y_schedule=[0], z_schedule=[0], scale_schedule=[1.0], detail_schedule=[100], randomness_schedule=[1], seed_schedule=[0], device="cuda"):
        voronoi = VoronoiNoise(width=width, height=height, scale=scale_schedule, detail=detail_schedule, seed=seed_schedule, 
                            X=x_schedule, Y=y_schedule, 
                            randomness=randomness_schedule, distance_metric=distance_metric, batch_size=batch_size, device=device)
        voronoi = voronoi.to(device)
        tensors = voronoi()
        return(tensors, batch_size)

```
