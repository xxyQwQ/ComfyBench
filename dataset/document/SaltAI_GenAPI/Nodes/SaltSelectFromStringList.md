# Select from String List
## Documentation
- Class name: `SaltSelectFromStringList`
- Category: `SALT/List`
- Output node: `False`

The `SaltSelectFromStringList` node enables the selection of an element from a list of strings based on a specified index. It abstracts the process of indexing into lists, providing a straightforward way to retrieve specific elements, and includes handling for out-of-range indices.
## Input types
### Required
- **`list_input`**
    - Represents the list of strings from which an element will be selected. It is crucial for determining the scope of selection.
    - Comfy dtype: `LIST`
    - Python dtype: `List[str]`
- **`index`**
    - Specifies the position in the list from which to select the element. It determines the exact element to be retrieved from the list.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`selection`**
    - Comfy dtype: `STRING`
    - The selected string from the input list, based on the provided index.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltSelectFromStringList:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "list_input": ("LIST",),
                "index": ("INT", {"default": 0, "min": 0}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("selection",)

    FUNCTION = "select"
    CATEGORY = f"{MENU_NAME}/List"

    def select(self, list_input, index):
        if index < 0 or index >= len(list_input):
            logger.warning(f"List index `{index}` out of range for length {len(list_input)}")
            index = -1
        return (str(list_input[index]),)

```
