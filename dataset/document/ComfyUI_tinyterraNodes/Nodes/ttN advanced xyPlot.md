---
tags:
- DataVisualization
- XYPlot
- XYPlotData
---

# advanced xyPlot
## Documentation
- Class name: `ttN advanced xyPlot`
- Category: `🌏 tinyterra/xyPlot`
- Output node: `False`

The `ttN advanced xyPlot` node is designed for advanced plotting capabilities, focusing on generating and manipulating XY plots. It enables the creation of plots based on X, Y, and optionally Z coordinates, with features such as flipping the X and Y axes, adjusting grid spacing, and saving individual plots. This node is tailored for visualizing complex data relationships through customizable 2D and 3D plots.
## Input types
### Required
- **`grid_spacing`**
    - Specifies the spacing between grid lines in the plot, affecting the plot's resolution and visual density.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`save_individuals`**
    - Determines whether to save each plotted point as an individual file, enabling detailed analysis of specific data points.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`flip_xy`**
    - Controls whether the X and Y axes should be swapped, allowing for alternative data perspective visualization.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`x_plot`**
    - The X coordinates for plotting, defining the horizontal positioning of data points.
    - Comfy dtype: `STRING`
    - Python dtype: `Optional[str]`
- **`y_plot`**
    - The Y coordinates for plotting, defining the vertical positioning of data points.
    - Comfy dtype: `STRING`
    - Python dtype: `Optional[str]`
- **`z_plot`**
    - The Z coordinates for plotting, adding a third dimension to the visualization when applicable.
    - Comfy dtype: `STRING`
    - Python dtype: `Optional[str]`
## Output types
- **`adv_xyPlot`**
    - Comfy dtype: `ADV_XYPLOT`
    - The generated XY plot data, including coordinates and configuration settings for visualization.
    - Python dtype: `Dict[str, Any]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ttN_advanced_XYPlot:
    version = '1.2.0'
    plotPlaceholder = "_PLOT\nExample:\n\n<axis number:label1>\n[node_ID:widget_Name='value']\n\n<axis number2:label2>\n[node_ID:widget_Name='value2']\n[node_ID:widget2_Name='value']\n[node_ID2:widget_Name='value']\n\netc..."

    def get_plot_points(plot_data, unique_id, plot_Line):
        if plot_data is None or plot_data.strip() == '':
            return None
        else:
            try:
                axis_dict = {}
                lines = plot_data.split('<')
                new_lines = []
                temp_line = ''

                for line in lines:
                    if line.startswith('lora'):
                        temp_line += '<' + line
                        new_lines[-1] = temp_line
                    else:
                        new_lines.append(line)
                        temp_line = line
                        
                for line in new_lines:
                    if line:
                        values_label = []
                        line = line.split('>', 1)
                        num, label = line[0].split(':', 1)
                        axis_dict[num] = {"label": label}
                        for point in line[1].split("']"):
                            if point.strip() == '':
                                continue
                            
                            node_id = point.split(':', 1)[0].split('[')[1]
                            axis_dict[num].setdefault(node_id, {})
                            input_name = point.split(':', 1)[1].split('=', 1)[0]
                            value = point.split("'", 1 )[1]
                            values_label.append((value, input_name, node_id))
                            
                            axis_dict[num][node_id][input_name] = value
                                
                        if label in ['v_label', 'tv_label', 'idtv_label']:
                            new_label = []
                            for value, input_name, node_id in values_label:
                                if label == 'v_label':
                                    new_label.append(value)
                                elif label == 'tv_label':
                                    new_label.append(f'{input_name}: {value}')
                                elif label == 'idtv_label':
                                    new_label.append(f'[{node_id}] {input_name}: {value}')
                            axis_dict[num]['label'] = ', '.join(new_label)
                        
            except ValueError:
                ttNl('Invalid Plot - defaulting to None...').t(f'advanced_XYPlot[{unique_id}] {plot_Line} Axis').warn().p()
                return None
            return axis_dict

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "grid_spacing": ("INT",{"min": 0, "max": 500, "step": 5, "default": 0,}),
                "save_individuals": ("BOOLEAN", {"default": False}),
                "flip_xy": ("BOOLEAN", {"default": False}),
                
                "x_plot": ("STRING",{"default": '', "multiline": True, "placeholder": 'X' + ttN_advanced_XYPlot.plotPlaceholder, "pysssss.autocomplete": False}),
                "y_plot": ("STRING",{"default": '', "multiline": True, "placeholder": 'Y' + ttN_advanced_XYPlot.plotPlaceholder, "pysssss.autocomplete": False}),
                "z_plot": ("STRING",{"default": '', "multiline": True, "placeholder": 'Z' + ttN_advanced_XYPlot.plotPlaceholder, "pysssss.autocomplete": False}),
            },
            "hidden": {
                "my_unique_id": "UNIQUE_ID",
                "ttNnodeVersion": ttN_advanced_XYPlot.version,
            },
        }

    RETURN_TYPES = ("ADV_XYPLOT", )
    RETURN_NAMES = ("adv_xyPlot", )
    FUNCTION = "plot"

    CATEGORY = "🌏 tinyterra/xyPlot"
    
    def plot(self, grid_spacing, save_individuals, flip_xy, x_plot=None, y_plot=None, z_plot=None, my_unique_id=None):
        x_plot = ttN_advanced_XYPlot.get_plot_points(x_plot, my_unique_id, 'X')
        y_plot = ttN_advanced_XYPlot.get_plot_points(y_plot, my_unique_id, 'Y')
        z_plot = ttN_advanced_XYPlot.get_plot_points(z_plot, my_unique_id, 'Z')

        if x_plot == {}:
            x_plot = None
        if y_plot == {}:
            y_plot = None

        if flip_xy == True:
            x_plot, y_plot = y_plot, x_plot

        xy_plot = {"x_plot": x_plot,
                   "y_plot": y_plot,
                   "z_plot": z_plot,
                   "grid_spacing": grid_spacing,
                   "save_individuals": save_individuals,}
        
        return (xy_plot, )

```
