# SystemNotification 🐍
## Documentation
- Class name: `SystemNotification_pysssss`
- Category: `utils`
- Output node: `True`

The SystemNotification node is designed to facilitate the creation and management of system notifications, allowing for customizable messages and modes of operation. It abstracts the complexity of handling system alerts, providing a straightforward way to notify users based on specific conditions or system states.
## Input types
### Required
- **`message`**
    - Specifies the default message to be displayed in the notification. This parameter allows for customization of the notification content, tailoring it to specific needs or contexts.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`any`**
    - Acts as a wildcard input, enabling flexible and dynamic handling of notifications without strict parameter constraints.
    - Comfy dtype: `*`
    - Python dtype: `AnyType`
- **`mode`**
    - Determines the conditions under which the notification is triggered, offering options for always displaying or conditionally based on the system's state.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `list[str]`
## Output types
- **`*`**
    - Comfy dtype: `*`
    - unknown
    - Python dtype: `unknown`
- **`ui`**
    - Provides a user interface component for displaying the notification, enhancing user interaction and feedback.
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SystemNotification:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "message": ("STRING", {"default": "Your notification has triggered."}),
            "any": (any, {}),
            "mode": (["always", "on empty queue"], {}),
        }}

    FUNCTION = "nop"
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)
    OUTPUT_NODE = True
    RETURN_TYPES = (any,)

    CATEGORY = "utils"

    def IS_CHANGED(self, **kwargs):
        return float("NaN")

    def nop(self, any, **kwargs):
        return {"ui": {"a": []}, "result": (any,)}

```
