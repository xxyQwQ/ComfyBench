---
tags:
- Audio
- SaltNodes
---

# Salt Workflow Input
## Documentation
- Class name: `SaltInput`
- Category: `SALT/IO`
- Output node: `True`

The SaltInput node is designed to facilitate the creation and management of inputs for workflows within the Salt AI platform. It allows for the specification of various input types, including strings, numbers, booleans, and files, and supports additional features such as image and mask inputs, setting allowed values, and specifying whether user overrides are required. This node plays a crucial role in customizing and controlling the flow of data into Salt AI workflows, ensuring that inputs are correctly defined and processed according to the workflow's requirements.
## Input types
### Required
- **`input_name`**
    - Specifies the name of the input. This name is used to identify the input within the workflow, making it easier to reference and manage.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`input_desc`**
    - Provides a description for the input. This description helps to clarify the purpose and expected content of the input, aiding in the workflow's documentation and usability.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`input_type`**
    - Defines the type of the input. This can range from basic types like strings and numbers to more complex types like images and files, allowing for versatile input handling in the workflow.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `list[str]`
- **`input_value`**
    - The actual value for the input. Depending on the input type, this could be text, a numerical value, an image path, etc., serving as the primary data that the workflow will process.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
- **`input_image`**
    - An optional image input. When provided, this allows the workflow to directly process an image, adding a visual data processing capability.
    - Comfy dtype: `IMAGE`
    - Python dtype: `Image`
- **`input_mask`**
    - An optional mask input. This is used in conjunction with image inputs to apply specific processing or transformations based on the mask.
    - Comfy dtype: `MASK`
    - Python dtype: `Mask`
- **`input_allowed_values`**
    - Specifies allowed values for the input. This is useful for inputs that should only accept a predefined set of values, ensuring data validity.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`user_override_required`**
    - Indicates whether a user override is required for the input. This can be used to enforce user interaction for certain inputs, ensuring that specific conditions are met before proceeding.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`relative_path`**
    - Determines whether the input value should be treated as a relative path. This is relevant for file inputs, affecting how the file path is interpreted and handled within the workflow.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`value`**
    - Comfy dtype: `*`
    - unknown
    - Python dtype: `unknown`
- **`ui`**
    - Returns a UI dictionary containing metadata and results related to the input processing. This includes information such as the input's unique ID, any relevant asset IDs, and the processed input data, facilitating integration with the Salt AI platform's UI components.
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltInput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_name": ("STRING", {}),
                "input_desc": ("STRING", {}),
                "input_type": (["STRING", "FLOAT", "INT", "BOOLEAN", "IMAGE", "MASK", "SEED", "FILE"],),
                "input_value": ("STRING", {"multiline": True, "dynamicPrompts": False})
            },
            "optional": {
                "input_image": ("IMAGE",),
                "input_mask": ("MASK",),
                "input_allowed_values": ("STRING", {"default": ""}),
                "user_override_required": ("BOOLEAN", {"default": False}),
                "relative_path": ("BOOLEAN", {"default": False})
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    OUTPUT_NODE = True
    RETURN_TYPES = (WILDCARD,)
    RETURN_NAMES = ("value",)

    FUNCTION = "input"
    CATEGORY = "SALT/IO"

    def input(
        self,
        input_name,
        input_desc,
        input_value,
        input_type,
        input_image=None,
        input_mask=None,
        input_allowed_values="",
        user_override_required=False,
        relative_path=False,
        unique_id=0,
    ):
        src_image = None
        src_file = None
        is_asset = False

        # Is an asset type
        if input_type in ["IMAGE", "MASK", "FILE"]:
            is_asset = True

        # UI Output
        ui = {
            "ui": {
                "salt_input": [{
                    "id": unique_id,
                    "name": input_name or "input_" + str(unique_id),
                    "description": input_desc or "",
                    "asset": is_asset or False,
                    "type": input_type or "string",
                    "value": input_value or "",
                }]
            }
        }

        out = ""
        if is_asset:
            # Input value must be evaluated first to override input images/masks
            if input_value.strip():
                input_value = input_value.strip()
                # Load image from path from input_value
                if input_value.endswith(('.png', '.jpeg', '.jpg', '.gif', '.webp', '.tiff')):
                    try:
                        src_image = Image.open(input_value).convert("RGBA")
                    except Exception as e:
                        errmsg = f"Error loading image from specified path {input_value}: {e}"
                        logger.warning(errmsg)
                # Passthrough input_value (which should be a path from Salt Backend)
                elif input_type == "FILE":

                    src_file = input_value # if os.path.exists(input_value) else "None"
                    if relative_path:
                        src_file = get_relative_path(src_file)

                    # Log value to console
                    log_values(unique_id, input_name, input_type, src_file)
                    
                    return {"ui": ui, "result": (src_file,)}
                else:
                    errmsg = "Invalid node configuration! Do you mean to use `IMAGE`, `MASK`, or `FILE` input_types?"
                    logger.error(errmsg)
                    raise AttributeError(errmsg)
            elif isinstance(input_image, torch.Tensor):
                # Input `IMAGE` is provided, so we act like a passthrough
                return (input_image, ui)
            elif isinstance(input_mask, torch.Tensor):
                # Input `MASK` is provided, so we act like a passthrough
                return (input_mask, ui)

            if src_image:
                if input_type == "MASK":
                    # If it's a mask and the image has an alpha channel, extract it
                    if src_image.mode == "RGBA":
                        alpha_channel = src_image.split()[-1]
                        src_image = pil2mask(alpha_channel.convert("L"))
                    # If no alpha channel, convert the whole image to grayscale as a mask (could be bitwise representation)
                    else:
                        src_image = pil2mask(src_image.convert("L"))
                elif input_type == "IMAGE":
                    # Ensure image is in RGB for `IMAGE`` data type
                    src_image = pil2tensor(src_image.convert("RGB"))

            else:
                # Gracefully allow execution to continue, provided a black image (to hopefully signal issue?)
                errmsg = "Unable to determine IMAGE or MASK to load!  Returning image blank"
                logger.warning(errmsg)

                src_blank = Image.new("RGB", (512, 512), (0, 0, 0))
                if input_type == "IMAGE":
                    src_image = pil2tensor(src_blank)
                else:
                    src_image = pil2mask(src_blank)

            # Log value to console
            log_values(unique_id, input_name, input_type, src_image)

            return (src_image, ui)

        # We're still here? We must be dealing with a primitive value
        if input_allowed_values != "" and input_value.strip() not in [o.strip() for o in input_allowed_values.split(',')]:
            errmsg = 'The provided input is not a supported value'
            logger.warning(errmsg)
            raise ValueError(errmsg)


        match input_type:
            case "STRING":
                out = str(input_value)
            case "INT":
                out = int(input_value)
            case "SEED":
                out = int(input_value)
            case "FLOAT":
                out = float(input_value)
            case "BOOLEAN":
                out = bool_str(input_value)
            case _:
                out = input_value

        # Log value to console
        log_values(unique_id, input_name, input_type, out)

        return (out, ui)

```
