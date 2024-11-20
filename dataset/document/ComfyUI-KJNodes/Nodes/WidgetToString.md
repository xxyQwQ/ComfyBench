# Widget To String
## Documentation
- Class name: `WidgetToString`
- Category: `KJNodes/text`
- Output node: `False`

The get_widget_value node is designed to select a specific widget within a node and output its value as a string. It enables dynamic access to widget data based on node ID and widget name, with an option to return all widget values from a node if required. This functionality is crucial for customizing user interactions and data retrieval within a workflow.
## Input types
### Required
- **`id`**
    - Specifies the unique identifier of the node whose widget's value is to be retrieved, essential for pinpointing the exact node within a workflow.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`widget_name`**
    - The name of the widget whose value is to be extracted as a string, critical for identifying the specific widget within the node.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`return_all`**
    - A boolean flag that determines whether the node should return the values of all widgets as a string, offering flexibility in the amount of information extracted.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`string`**
    - Comfy dtype: `STRING`
    - The extracted widget value(s) converted to a string, essential for presenting or further processing the widget data in text form.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class WidgetToString:
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "id": ("INT", {"default": 0}),
                "widget_name": ("STRING", {"multiline": False}),
                "return_all": ("BOOLEAN", {"default": False}),
            },
            
            "hidden": {"extra_pnginfo": "EXTRA_PNGINFO",
                       "prompt": "PROMPT"},
        }

    RETURN_TYPES = ("STRING", )
    FUNCTION = "get_widget_value"
    CATEGORY = "KJNodes/text"
    DESCRIPTION = """
Selects a node and it's specified widget and outputs the value as a string.  
To see node id's, enable node id display from Manager badge menu.
"""

    def get_widget_value(self, id, widget_name, extra_pnginfo, prompt, return_all=False):
        workflow = extra_pnginfo["workflow"]
        results = []
        for node in workflow["nodes"]:
            node_id = node["id"]

            if node_id != id:
                continue

            values = prompt[str(node_id)]
            if "inputs" in values:
                if return_all:
                    results.append(', '.join(f'{k}: {str(v)}' for k, v in values["inputs"].items()))
                elif widget_name in values["inputs"]:
                    v = str(values["inputs"][widget_name])  # Convert to string here
                    return (v, )
                else:
                    raise NameError(f"Widget not found: {id}.{widget_name}")
        if not results:
            raise NameError(f"Node not found: {id}")
        return (', '.join(results).strip(', '), )

```
