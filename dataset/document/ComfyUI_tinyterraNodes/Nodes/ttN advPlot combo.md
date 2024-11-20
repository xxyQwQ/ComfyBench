# advPlot combo
## Documentation
- Class name: `ttN advPlot combo`
- Category: `🌏 tinyterra/xyPlot`
- Output node: `True`

The node ttN_advPlot_combo is designed for generating advanced plot text outputs based on a combination of input parameters such as node identifiers, widget types, and mode selections. It dynamically constructs plot text by mapping values to specified labels and formatting them for display, supporting customization of plot elements through various configurations including range and selection modes.
## Input types
### Required
- **`node`**
    - Specifies the node identifier used in the plot text, essential for linking the plot elements to their respective nodes.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`widget`**
    - Defines the widget type associated with the plot values, playing a crucial role in how values are presented and interacted with in the plot.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`mode`**
    - Determines the mode of value selection for plotting, such as 'all', 'range', or 'select', affecting how values are gathered and displayed.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`start_from`**
    - Specifies the starting point for 'range' mode, indicating the beginning of the value range to be plotted.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`end_with`**
    - Defines the end point for 'range' mode, marking the conclusion of the value range to be plotted.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`select`**
    - Indicates the widget used for 'select' mode, allowing specific values to be chosen for plotting.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`selection`**
    - Contains the selected values for plotting in 'select' mode, serving as a direct input of plot data.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`label_type`**
    - Determines the labeling scheme for the plot elements, influencing the textual representation and organization of plot data.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`plot_text`**
    - Comfy dtype: `STRING`
    - The generated plot text, encapsulating the formatted plot elements based on the input parameters.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ttN_advPlot_combo:
    version = '1.0.0'
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "node": ([AnyType("Connect to xyPlot for options"),],{}),
                "widget": ([AnyType("Select node for options"),],{}),

                "mode": (['all', 'range', 'select'],),
                "start_from": ([AnyType("Select widget for options"),],),
                "end_with": ([AnyType("Select widget for options"),],),
                
                "select": ([AnyType("Select widget for options"),],),
                "selection": ("STRING", {"default":"","multiline": True}),
                
                "label_type": (['Values', 'Title and Values', 'ID, Title and Values'],{"default": "Values"}),
            },
            "hidden": {
                "ttNnodeVersion": ttN_advPlot_range.version, "prompt": "PROMPT",
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("plot_text",)
    FUNCTION = "plot"
    OUTPUT_NODE = True

    CATEGORY = "🌏 tinyterra/xyPlot"

    def plot(self, node, widget, mode, start_from, end_with, select, selection, label_type, prompt=None):
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

        class_type = prompt[nodeid]['class_type']
        class_def = nodes.NODE_CLASS_MAPPINGS[class_type]
        valid_inputs = class_def.INPUT_TYPES()
        options = valid_inputs["required"][widget][0] or valid_inputs["optional"][widget][0]

        vals = []
        if mode == 'all':
            vals = options
        elif mode == 'range':
            start_index = options.index(start_from)
            stop_index = options.index(end_with) + 1
            if start_index > stop_index:
                start_index, stop_index = stop_index, start_index
            vals = options[start_index:stop_index]
        elif mode == 'select':
            selection = selection.split('\n')
            for s in selection:
                s.strip()
                if s in options:
                    vals.append(s)

        for i, val in enumerate(vals):
            line = f"[{nodeid}:{widget}='{val}']"
            plot_text.append(f"<{i+1}:{label}>")
            plot_text.append(line)
            
        out = '\n'.join(plot_text)

        return {"ui": {"text": out}, "result": (out,)}

```
