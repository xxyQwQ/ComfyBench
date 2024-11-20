---
tags:
- ConditionalSelection
- ImpactPack
---

# ImpactLogger
## Documentation
- Class name: `ImpactLogger`
- Category: `ImpactPack/Debug`
- Output node: `True`

The ImpactLogger node is designed for logging and debugging purposes within the ImpactPack framework. It captures and prints data, text, and additional information related to the execution of nodes, facilitating the monitoring and troubleshooting of workflows.
## Input types
### Required
- **`data`**
    - The primary data input for logging. This can be of any type and is central to what the logger will output, showcasing the node's flexibility in handling various data forms.
    - Comfy dtype: `*`
    - Python dtype: `object`
- **`text`**
    - A text input that allows for multiline strings, providing context or additional information to be logged alongside the data.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
The node doesn't have output types
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ImpactLogger:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "data": (any_typ, ""),
                        "text": ("STRING", {"multiline": True}),
                    },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "unique_id": "UNIQUE_ID"},
                }

    CATEGORY = "ImpactPack/Debug"

    OUTPUT_NODE = True

    RETURN_TYPES = ()
    FUNCTION = "doit"

    def doit(self, data, text, prompt, extra_pnginfo, unique_id):
        shape = ""
        if hasattr(data, "shape"):
            shape = f"{data.shape} / "

        print(f"[IMPACT LOGGER]: {shape}{data}")

        print(f"         PROMPT: {prompt}")

        # for x in prompt:
        #     if 'inputs' in x and 'populated_text' in x['inputs']:
        #         print(f"PROMPT: {x['10']['inputs']['populated_text']}")
        #
        # for x in extra_pnginfo['workflow']['nodes']:
        #     if x['type'] == 'ImpactWildcardProcessor':
        #         print(f" WV : {x['widgets_values'][1]}\n")

        PromptServer.instance.send_sync("impact-node-feedback", {"node_id": unique_id, "widget_name": "text", "type": "TEXT", "value": f"{data}"})
        return {}

```
