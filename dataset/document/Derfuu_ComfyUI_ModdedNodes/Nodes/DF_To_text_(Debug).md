---
tags:
- Text
---

# To text (Debug)
## Documentation
- Class name: `DF_To_text_(Debug)`
- Category: `Derfuu_Nodes/Debug`
- Output node: `True`

The `DF_To_text_(Debug)` node is designed for debugging purposes, allowing users to log and inspect any data passed through it. It converts the input data to a string format, logs the original and string-converted data for debugging, and handles exceptions by logging them. This node facilitates the observation and troubleshooting of data flow within node-based processing pipelines.
## Input types
### Required
- **`ANY`**
    - Accepts any type of data for debugging purposes. It logs the input data in its original form and after conversion to a string, aiding in the inspection and troubleshooting of data flow.
    - Comfy dtype: `*`
    - Python dtype: `Any`
## Output types
- **`SAME AS INPUT`**
    - Comfy dtype: `*`
    - Returns the original input data, allowing it to be passed through for further processing or inspection.
    - Python dtype: `Any`
- **`STRING`**
    - Comfy dtype: `STRING`
    - Returns the input data converted to a string, or an error message if an exception occurs, facilitating debugging and error tracking.
    - Python dtype: `str`
- **`ui`**
    - Provides a user interface component displaying the debug information as text. It shows the input data in both its original and string-converted forms, or an error message if an exception occurs.
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ShowDataDebug:
    CATEGORY = TREE_DEBUG

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ANY": Field.any(),
            },
        }

    RETURN_TYPES = (ANY, "STRING", )
    RETURN_NAMES = ("SAME AS INPUT", "STRING", )
    OUTPUT_NODE = True
    IS_CHANGED = True
    FUNCTION = "func"

    def func(self, ANY = None):
        out = ANY
        try:
            out = str(out)
            logging.info(colorize(f"[DEBUG]: {ANY}", ConsoleColor.blue.value))
        except Exception as e:
            logging.info(colorize(f"[DEBUG-EXCEPTION]: {e}", ConsoleColor.bold_red.value))
            out = str(e)
        return {"ui": {"text": [out]}, "result": (ANY, out)}

```
