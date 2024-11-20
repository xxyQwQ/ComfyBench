# advPlot string
## Documentation
- Class name: `ttN advPlot string`
- Category: `🌏 tinyterra/xyPlot`
- Output node: `True`

This node is designed to generate advanced plotting strings based on specified parameters, including node identification, widget interaction, and value formatting. It dynamically constructs plot text elements that can be used to visualize data or control elements in a UI context, emphasizing flexibility in label presentation and numerical value adjustment.
## Input types
### Required
- **`node`**
    - Identifies the node to which the plot string will be associated, enabling dynamic interaction with xyPlot options.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`widget`**
    - Specifies the widget to interact with, allowing for dynamic selection of options based on the node's configuration.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`text`**
    - The text input that will be split and formatted according to the delimiter to generate plot values.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`delimiter`**
    - Defines the character or sequence used to split the text input into individual plot values.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`label_type`**
    - Determines the type of label to be used for each plot value, affecting the presentation and information included.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`plot_text`**
    - Comfy dtype: `STRING`
    - The generated plot text, ready for use in visualization or UI control.
    - Python dtype: `str`
- **`ui`**
    - Provides the generated plot text in a format suitable for UI display, encapsulating the visualization logic.
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ttN_advPlot_string:
    version = '1.0.0'
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "node": ([AnyType("Connect to xyPlot for options"),],{}),
                "widget": ([AnyType("Select node for options"),],{}),

                "text": ("STRING", {"default":"","multiline": True}),
                "delimiter": ("STRING", {"default":"\\n","multiline": False}),
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

    def plot(self, node, widget, text, delimiter, label_type):
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
        
        delimiter = delimiter.replace('\\n', '\n')
        vals = text.split(delimiter)

        for i, val in enumerate(vals):
            line = f"[{nodeid}:{widget}='{val}']"
            plot_text.append(f"<{i+1}:{label}>")
            plot_text.append(line)
            
        out = '\n'.join(plot_text)

        return {"ui": {"text": out}, "result": (out,)}

```
