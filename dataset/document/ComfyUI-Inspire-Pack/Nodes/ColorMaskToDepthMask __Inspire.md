# Color Mask To Depth Mask (Inspire)
## Documentation
- Class name: `ColorMaskToDepthMask __Inspire`
- Category: `InspirePack/Regional`
- Output node: `False`

The `ColorMaskToDepthMask` node is designed to convert a color mask into a depth mask based on various specifications including color codes, base values, dilation, and flatten methods. This process involves analyzing the color mask to apply transformations that result in a depth representation, taking into account the specified parameters to adjust the depth effect.
## Input types
### Required
- **`color_mask`**
    - `color_mask` is the input tensor representing the color mask to be converted. It plays a crucial role in determining which pixels will be considered for conversion based on the specified mask color.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`spec`**
    - `spec` defines the specifications for converting color codes into depth values, including the color code and its corresponding configuration.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`base_value`**
    - `base_value` sets the initial depth value for the mask before applying any color-specific transformations.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`dilation`**
    - `dilation` specifies the amount by which to dilate the mask, affecting the depth representation by expanding or contracting the mask areas.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`flatten_method`**
    - `flatten_method` determines how multiple mask layers are combined into a single mask, with options such as 'override', 'max', and 'sum'.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`mask`**
    - Comfy dtype: `MASK`
    - The output is a depth mask tensor, representing the transformed depth mask derived from the original color mask and the specified transformations.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ColorMaskToDepthMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "color_mask": ("IMAGE",),
                "spec": ("STRING", {"multiline": True, "default": "#FF0000:1.0\n#000000:1.0"}),
                "base_value": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
                "dilation": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),
                "flatten_method": (["override", "sum", "max"],),
            },
        }

    RETURN_TYPES = ("MASK", )
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Regional"

    def doit(self, color_mask, spec, base_value, dilation, flatten_method):
        specs = spec.split('\n')
        pat = re.compile("(?P<color_code>#[A-F0-9]+):(?P<cfg>[0-9]+(.[0-9]*)?)")

        masks = [torch.ones((1, color_mask.shape[1], color_mask.shape[2])) * base_value]
        for x in specs:
            match = pat.match(x)
            if match:
                mask = color_to_mask(color_mask=color_mask, mask_color=match['color_code']) * float(match['cfg'])
                mask = utils.dilate_mask(mask, dilation)
                masks.append(mask)

        if masks:
            masks = torch.cat(masks, dim=0)
            if flatten_method == 'override':
                masks = utils.flatten_non_zero_override(masks)
            elif flatten_method == 'max':
                masks = torch.max(masks, dim=0)[0]
            else:  # flatten_method == 'sum':
                masks = torch.sum(masks, dim=0)

            masks = torch.clamp(masks, min=0.0, max=1.0)
            masks = masks.unsqueeze(0)
        else:
            masks = torch.tensor([])

        return (masks, )

```
