---
tags:
- DataTypeAgnostic
- Debugging
---

# Show Any
## Documentation
- Class name: `easy showAnything`
- Category: `EasyUse/Logic`
- Output node: `True`

This node is designed to display any given input in a user-friendly manner, regardless of its original data type. It abstracts the complexity of data types, making it easier for users to visualize and understand the data presented within the ComfyUI environment.
## Input types
### Required
### Optional
- **`anything`**
    - Accepts any data type as input, allowing for a wide range of data to be displayed. This flexibility ensures that users can visualize virtually any piece of data without concern for compatibility issues.
    - Comfy dtype: `*`
    - Python dtype: `Any`
## Output types
The node doesn't have output types
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class showAnything:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}, "optional": {"anything": (AlwaysEqualProxy("*"), {}), },
                "hidden": {"unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO",
            }}

    RETURN_TYPES = ()
    INPUT_IS_LIST = True
    OUTPUT_NODE = True
    FUNCTION = "log_input"
    CATEGORY = "EasyUse/Logic"

    def log_input(self, unique_id=None, extra_pnginfo=None, **kwargs):

        values = []
        if "anything" in kwargs:
            for val in kwargs['anything']:
                try:
                    if type(val) is str:
                        values.append(val)
                    else:
                        val = json.dumps(val)
                        values.append(str(val))
                except Exception:
                    values.append(str(val))
                    pass

        if not extra_pnginfo:
            print("Error: extra_pnginfo is empty")
        elif (not isinstance(extra_pnginfo[0], dict) or "workflow" not in extra_pnginfo[0]):
            print("Error: extra_pnginfo[0] is not a dict or missing 'workflow' key")
        else:
            workflow = extra_pnginfo[0]["workflow"]
            node = next((x for x in workflow["nodes"] if str(x["id"]) == unique_id[0]), None)
            if node:
                node["widgets_values"] = [values]

        return {"ui": {"text": values}}

```
