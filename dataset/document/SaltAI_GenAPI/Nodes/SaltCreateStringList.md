# Create String List
## Documentation
- Class name: `SaltCreateStringList`
- Category: `SALT/String`
- Output node: `False`

This node is designed to create a list containing a single string element. It serves as a utility for initializing a list structure with a specified string, facilitating operations that require list input formats.
## Input types
### Required
- **`string_input`**
    - The string to be initialized into a list. This parameter is crucial for defining the sole element of the newly created list, setting the foundation for further list-based manipulations.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`string_list`**
    - Comfy dtype: `LIST`
    - The output is a list containing the initial string input as its only element. This list is ready for further processing or manipulation in subsequent operations.
    - Python dtype: `List[str]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltCreateStringList:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string_input": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("string_list",)
    OUTPUT_IS_LIST = (True,)

    FUNCTION = "create_list"
    CATEGORY = f"{MENU_NAME}/String"

    def create_list(self, string_input):
        return ([string_input],)

```
