---
tags:
- Debugging
---

# Debug (mtb)
## Documentation
- Class name: `Debug (mtb)`
- Category: `mtb/debug`
- Output node: `True`

The MTB_Debug node is designed for experimental debugging of various Comfy values, with plans to support additional types and widgets in the future. It allows for the inspection and logging of data passed through it, facilitating the debugging process by providing insights into the data's structure and content.
## Input types
### Required
- **`output_to_console`**
    - Determines whether the debug output should be printed to the console. Enabling this option allows for immediate visual inspection of the data being debugged, aiding in the debugging process.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`ui`**
    - Provides a structured output containing base64-encoded images and text, which can be used for debugging purposes by visually inspecting the processed data.
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class MTB_Debug:
    """Experimental node to debug any Comfy values.

    support for more types and widgets is planned.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"output_to_console": ("BOOLEAN", {"default": False})},
        }

    RETURN_TYPES = ()
    FUNCTION = "do_debug"
    CATEGORY = "mtb/debug"
    OUTPUT_NODE = True

    def do_debug(self, output_to_console: bool, **kwargs):
        output = {
            "ui": {"b64_images": [], "text": []},
            # "result": ("A"),
        }

        processors = {
            torch.Tensor: process_tensor,
            list: process_list,
            dict: process_dict,
            bool: process_bool,
        }
        if output_to_console:
            for k, v in kwargs.items():
                log.info(f"{k}: {v}")

        for anything in kwargs.values():
            processor = processors.get(type(anything), process_text)

            processed_data = processor(anything)

            for ui_key, ui_value in processed_data.items():
                output["ui"][ui_key].extend(ui_value)

        return output

```
