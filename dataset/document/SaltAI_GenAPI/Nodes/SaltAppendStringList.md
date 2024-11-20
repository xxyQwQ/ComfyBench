# Append to String List
## Documentation
- Class name: `SaltAppendStringList`
- Category: `SALT/String`
- Output node: `False`

The SaltAppendStringList node is designed to extend a list of strings by appending a new string to it. This functionality is essential for dynamically growing lists in applications that require concatenation or accumulation of string data.
## Input types
### Required
- **`string_list`**
    - The list of strings to which a new string will be appended. It serves as the base list that will be modified by adding another string to its end.
    - Comfy dtype: `LIST`
    - Python dtype: `List[str]`
- **`string_input`**
    - The string to be appended to the list. This parameter represents the new data being added to the existing list of strings.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`updated_list`**
    - Comfy dtype: `LIST`
    - The list of strings after the specified string has been appended to it. This reflects the updated state of the list, showcasing the addition of new string data.
    - Python dtype: `List[str]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltAppendStringList:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string_list": ("LIST", {"forceInput": True}),
                "string_input": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("updated_list",)
    OUTPUT_IS_LIST = (True,)

    FUNCTION = "append_list"
    CATEGORY = f"{MENU_NAME}/String"

    def append_list(self, string_list, string_input):
        string_list.append(string_input)
        return (string_list,)

```
