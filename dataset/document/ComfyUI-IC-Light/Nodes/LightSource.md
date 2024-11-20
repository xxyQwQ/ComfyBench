# Simple Light Source
## Documentation
- Class name: `LightSource`
- Category: `IC-Light`
- Output node: `False`

The LightSource node generates gradient images simulating various lighting effects based on specified parameters. It allows for the creation of dynamic lighting conditions in images, enhancing visual depth and realism.
## Input types
### Required
- **`light_position`**
    - Specifies the position of the simulated light source within the image, affecting the direction and spread of the light gradient.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `Enum['LightPosition']`
- **`multiplier`**
    - Adjusts the intensity and spread of the light gradient, enabling finer control over the lighting effect.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`start_color`**
    - Defines the starting color of the light gradient, allowing for customization of the light's appearance.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`end_color`**
    - Sets the ending color of the light gradient, facilitating the creation of varied lighting effects.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`width`**
    - Determines the width of the generated gradient image, impacting the overall size of the light effect.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - Specifies the height of the generated gradient image, influencing the scale of the light effect.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`batch_size`**
    - Controls the number of gradient images generated in a batch, supporting batch processing of light effects.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`prev_image`**
    - Optionally includes a previous image to be combined with the generated light gradient, enhancing composite image creation.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
## Output types
- **`IMAGE`**
    - Comfy dtype: `IMAGE`
    - Outputs a gradient image simulating the specified light source effect, ready for further processing or visualization.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LightSource:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "light_position": ([member.value for member in LightPosition],),
                "multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.001}),
                "start_color": ("STRING", {"default": "#FFFFFF"}),
                "end_color": ("STRING", {"default": "#000000"}),
                "width": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8, }),
                "height": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8, }),
                },
            "optional": {
                "batch_size": ("INT", { "default": 1, "min": 1, "max": 4096, "step": 1, }),
                "prev_image": ("IMAGE",),
                } 
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "IC-Light"
    DESCRIPTION = """
Generates a gradient image that can be used  
as a simple light source.  The color can be  
specified in RGB or hex format.  
"""

    def execute(self, light_position, multiplier, start_color, end_color, width, height, batch_size=1, prev_image=None):
        def toRgb(color):
            if color.startswith('#') and len(color) == 7:  # e.g. "#RRGGBB"
                color_rgb =tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
            else:  # e.g. "255,255,255"
                color_rgb = tuple(int(i) for i in color.split(','))
            return color_rgb
        lightPosition = LightPosition(light_position)
        start_color_rgb = toRgb(start_color)
        end_color_rgb = toRgb(end_color)
        image = generate_gradient_image(width, height, start_color_rgb, end_color_rgb, multiplier, lightPosition)
        
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        image = image.repeat(batch_size, 1, 1, 1)
        if prev_image is not None:
            image = torch.cat((prev_image, image), dim=0)
        return (image,)

```
