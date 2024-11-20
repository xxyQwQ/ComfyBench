# advPlot range
## Documentation
- Class name: `ttN advPlot range`
- Category: `🌏 tinyterra/xyPlot`
- Output node: `True`

The node ttN_advPlot_range is designed to generate a range of values for plotting purposes, providing a flexible way to create customized plot labels and values based on specified parameters. It supports different modes of range generation, including step-based and numerical step intervals, and allows for the inclusion of custom labels, making it suitable for a wide array of plotting scenarios.
## Input types
### Required
- **`node`**
    - Connects to an xyPlot node, determining the context and options for the range generation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`widget`**
    - Selects a specific widget from the connected node to apply the range generation, influencing the plot's output.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`range_mode`**
    - Defines the method of range generation, such as step-based or numerical steps, affecting how values are calculated.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
- **`start`**
    - Specifies the starting value of the range, serving as the initial point from which the range generation begins.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`step`**
    - Determines the increment between each value in the range, allowing for precise control over the distribution of values.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`stop`**
    - Defines the ending value of the range, marking the limit up to which values are generated.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`include_stop`**
    - Determines whether the stop value is included in the range, allowing for precise control over the range's boundaries.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`num_steps`**
    - Indicates the number of steps or intervals within the specified range, enabling detailed customization of the range's granularity.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`label_type`**
    - Specifies the type of label to be used for each value in the range, offering options for different levels of detail in labeling.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
## Output types
- **`plot_text`**
    - Comfy dtype: `STRING`
    - Generates a textual representation of the range values with labels, suitable for display or further processing.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ttN_advPlot_range:
    version = '1.1.0'
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "node": ([AnyType("Connect to xyPlot for options"),],{}),
                "widget": ([AnyType("Select node for options"),],{}),

                "range_mode": (['step_int','num_steps_int','step_float','num_steps_float'],{}),
                "start": ("FLOAT", {"min": -0xffffffffffffffff, "max": 0xffffffffffffffff, "step": 0.01, "default": 1,}),
                "step": ("FLOAT", {"min": -0xffffffffffffffff, "max": 0xffffffffffffffff, "step": 0.01, "default": 1,}),
                "stop": ("FLOAT", {"min": -0xffffffffffffffff, "max": 0xffffffffffffffff, "step": 0.01, "default": 5,}),
                "include_stop": ("BOOLEAN",{"default": True}),
                "num_steps": ("INT", {"min": 1, "max": 1000, "step": 1, "default": 5,}),

                "label_type": (['Values', 'Title and Values', 'ID, Title and Values'],{"default": "Values"}),

            },
            "hidden": {
                "ttNnodeVersion": ttN_advPlot_range.version,
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("plot_text",)
    FUNCTION = "plot"
    OUTPUT_NODE = True

    CATEGORY = "🌏 tinyterra/xyPlot"

    def plot(self, node, widget, range_mode, start, step, stop, include_stop, num_steps, label_type):
        if '[' in node and ']' in node:
            nodeid = node.split('[', 1)[1].split(']', 1)[0]
        else:
            return {"ui": {"text": ''}, "result": ('',)}
        
        label_map = {
            'Values': 'v_label',
            'Title and Values': 'tv_label',
            'ID, Title and Values': 'idtv_label',
        }
        label = label_map[label_type]

        plot_text = []
        vals = []
        
        if range_mode.startswith('step_'):
            for num in range(1, num_steps + 1):
                vals.append(start + step * (num - 1))
        if range_mode.startswith('num_steps'):
            vals = np.linspace(start, stop, num_steps, endpoint=include_stop).tolist()

        for i, val in enumerate(vals):
            if range_mode.endswith('int'):
                val = int(round(val, 0))
            else:
                val = round(val, 2)
            line = f"[{nodeid}:{widget}='{val}']"
            plot_text.append(f"<{i+1}:{label}>")
            plot_text.append(line)
            
        out = '\n'.join(plot_text)

        return {"ui": {"text": out}, "result": (out,)}

```
