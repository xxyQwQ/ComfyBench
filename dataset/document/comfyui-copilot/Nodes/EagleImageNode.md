---
tags:
- Image
---

# Eagle Image Node for PNGInfo
## Documentation
- Class name: `EagleImageNode`
- Category: `copilot`
- Output node: `False`

The EagleImageNode is designed for processing and handling image sequences, particularly focusing on adjusting images to a uniform size, converting them to a consistent format, and preparing them for further processing or analysis. It emphasizes the manipulation of image data, including resizing, format conversion, and mask generation, to facilitate downstream tasks such as image composition or analysis.
## Input types
### Required
- **`image`**
    - Specifies the image to be processed, essential for performing operations such as resizing, format conversion, and mask generation.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`int_field`**
    - An integer parameter that influences the node's processing logic, such as defining the dimensions for resizing or the number of iterations for certain operations, directly impacting the output's characteristics.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`float_field`**
    - A floating-point parameter that may adjust the intensity or threshold levels of certain image processing functions, affecting the visual outcome of the processed image.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`print_to_screen`**
    - A boolean parameter that controls the verbosity of the node's operation, determining whether intermediate or final processing results are displayed to the user.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `bool`
- **`string_field`**
    - A string parameter that could specify file paths, format types, or other textual configurations relevant to the node's processing tasks.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
The node doesn't have output types
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class EagleImageNode(BaseNode):
    """
    A example node

    Class methods
    -------------
    INPUT_TYPES (dict):
        Tell the main program input parameters of nodes.

    Attributes
    ----------
    RETURN_TYPES (`tuple`):
        The type of each element in the output tulple.
    RETURN_NAMES (`tuple`):
        Optional: The name of each output in the output tulple.
    FUNCTION (`str`):
        The name of the entry-point method. For example, if `FUNCTION = "execute"` then it will run Example().execute()
    OUTPUT_NODE ([`bool`]):
        If this node is an output node that outputs a result/image from the graph. The SaveImage node is an example.
        The backend iterates on these output nodes and tries to execute all their parents if their parent graph is properly connected.
        Assumed to be False if not present.
    CATEGORY (`str`):
        The category the node should appear in the UI.
    execute(s) -> tuple || None:
        The entry point method. The name of this method must be the same as the value of property `FUNCTION`.
        For example, if `FUNCTION = "execute"` then this method's name must be `execute`, if `FUNCTION = "foo"` then it must be `foo`.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        """
        Return a dictionary which contains config for all input fields.
        Some types (string): "MODEL", "VAE", "CLIP", "CONDITIONING", "LATENT", "IMAGE", "INT", "STRING", "FLOAT".
        Input types "INT", "STRING" or "FLOAT" are special values for fields on the node.
        The type can be a list for selection.

        Returns: `dict`:
            - Key input_fields_group (`string`): Can be either required, hidden or optional. A node class must have property `required`
            - Value input_fields (`dict`): Contains input fields config:
                * Key field_name (`string`): Name of a entry-point method's argument
                * Value field_config (`tuple`):
                    + First value is a string indicate the type of field or a list for selection.
                    + Secound value is a config for type "INT", "STRING" or "FLOAT".
        """
        return {
            "required": {
                "image": ("IMAGE",),
                "int_field": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,  # Minimum value
                        "max": 4096,  # Maximum value
                        "step": 64,  # Slider's step
                    },
                ),
                "float_field": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01},
                ),
                "print_to_screen": (["enable", "disable"],),
                "string_field": (
                    "STRING",
                    {
                        "multiline": False,
                        # True if you want the field to look like the one on the ClipTextEncode node
                        "default": "Hello World!",
                    },
                ),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    # RETURN_NAMES = ("image_output_name",)

    FUNCTION = "do_saving"

    def do_saving(
            self,
            image,
            string_field,
            int_field,
            float_field,
            print_to_screen,
            prompt=None,
            extra_pnginfo=None,
    ):
        print("--->imgae", "prompt", prompt, "extra", extra_pnginfo)
        if print_to_screen == "enable":
            print(
                f"""Your input contains:
                string_field aka input text: {string_field}
                int_field: {int_field}
                float_field: {float_field}
            """
            )
        # do some processing on the image, in this example I just invert it
        a = parse_workflow(prompt)
        print(a)

```
