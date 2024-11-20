---
tags:
- Noise
- NoisePatterns
---

# Scheduled Perline Power Fractal Generator
## Documentation
- Class name: `SaltScheduledPerlinPowerFractalNoise`
- Category: `SALT/AudioViz/Scheduling/Image`
- Output node: `False`

This node is designed to generate complex, visually appealing noise patterns by leveraging the Perlin noise algorithm. It intricately schedules and modifies Perlin noise parameters over time or across frames to produce dynamic, fractal-based noise textures that can be used for a variety of visual effects in audio visualization or graphical applications.
## Input types
### Required
- **`batch_size`**
    - Specifies the number of noise patterns to generate in a single batch, allowing for efficient bulk processing of noise textures.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`width`**
    - Determines the width of the generated noise texture, defining the horizontal resolution of the output pattern.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - Sets the height of the generated noise texture, establishing the vertical resolution of the output pattern.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`scale_schedule`**
    - A schedule of scale values to apply to the noise generation process, enabling dynamic control over the zoom level of the noise textures across different frames or instances.
    - Comfy dtype: `LIST`
    - Python dtype: `List[float]`
- **`octaves_schedule`**
    - Controls the level of detail in the noise texture by specifying a schedule for the number of octaves to use, affecting the complexity and richness of the generated patterns.
    - Comfy dtype: `LIST`
    - Python dtype: `List[int]`
- **`persistence_schedule`**
    - Adjusts the amplitude of each octave in the noise texture according to a predefined schedule, influencing the contrast between high and low-frequency details.
    - Comfy dtype: `LIST`
    - Python dtype: `List[float]`
- **`lacunarity_schedule`**
    - Defines a schedule for the lacunarity parameter, which affects the gap between successive octaves in the noise texture, thereby influencing its overall roughness and texture.
    - Comfy dtype: `LIST`
    - Python dtype: `List[float]`
- **`exponent_schedule`**
    - Specifies a schedule for the exponent values applied to the noise, allowing for fine-tuned adjustments to the intensity and distribution of the noise texture.
    - Comfy dtype: `LIST`
    - Python dtype: `List[float]`
- **`seed_schedule`**
    - A sequence of seed values to initialize the noise generation process, ensuring variability and uniqueness across different noise textures.
    - Comfy dtype: `LIST`
    - Python dtype: `List[int]`
- **`clamp_min_schedule`**
    - Sets a schedule for the minimum clamping value, defining the lower bound for the noise texture values to ensure they stay within a desired range.
    - Comfy dtype: `LIST`
    - Python dtype: `List[float]`
- **`clamp_max_schedule`**
    - Determines a schedule for the maximum clamping value, setting the upper limit for the noise texture values to maintain control over the output intensity.
    - Comfy dtype: `LIST`
    - Python dtype: `List[float]`
- **`device`**
    - Specifies the computing device (CPU or GPU) on which the noise generation process will be executed, affecting performance and efficiency.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`images`**
    - Comfy dtype: `IMAGE`
    - The generated noise images as a tensor, ready for further processing or visualization in various applications.
    - Python dtype: `torch.Tensor`
- **`batch_size`**
    - Comfy dtype: `INT`
    - The number of noise textures generated, corresponding to the input batch size, facilitating batch processing and management.
    - Python dtype: `int`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class SaltScheduledPerlinPowerFractalNoise:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch_size": ("INT", {"min": 1, "max": 4096}),
                "width": ("INT", {"default": 256, "min": 64, "max": MAX_RESOLUTION}),
                "height": ("INT", {"default": 256, "min": 64, "max": MAX_RESOLUTION}),
            },
            "optional": {
                "scale_schedule": ("LIST",),
                "octaves_schedule": ("LIST",),
                "persistence_schedule": ("LIST",),
                "lacunarity_schedule": ("LIST",),
                "exponent_schedule": ("LIST",),
                "seed_schedule": ("LIST",),
                "clamp_min_schedule": ("LIST",),
                "clamp_max_schedule": ("LIST",),
                "device": (["cuda", "cpu"],),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "batch_size")

    FUNCTION = "generate"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Scheduling/Image"

    def generate(self, batch_size, width, height, scale_schedule=[1.0], octaves_schedule=[4], persistence_schedule=[0.5], lacunarity_schedule=[2.0], exponent_schedule=[1.0], seed_schedule=[0], clamp_min_schedule=[-0.5], clamp_max_schedule=[1.5], device="cuda"):
        octaves_schedule = [int(octave) for octave in octaves_schedule]
        ppfn = PerlinPowerFractalNoise(
            width, height, 
            scale=scale_schedule, 
            octaves=octaves_schedule, 
            persistence=persistence_schedule, 
            lacunarity=lacunarity_schedule, 
            exponent=exponent_schedule, 
            seeds=seed_schedule, 
            clamp_min=clamp_min_schedule, 
            clamp_max=clamp_max_schedule, 
            batch_size=batch_size, 
            device=device
        )
        noise_tensor = ppfn.forward()
        return (noise_tensor, batch_size)

```
