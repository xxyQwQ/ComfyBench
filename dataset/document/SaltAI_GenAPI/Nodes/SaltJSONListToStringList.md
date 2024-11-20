# JSON List to Iterative List
## Documentation
- Class name: `SaltJSONListToStringList`
- Category: `SALT/JSON`
- Output node: `False`

This node is designed to convert a JSON list into a list of strings, ensuring that the input JSON is properly formatted and is indeed a list before proceeding with the conversion.
## Input types
### Required
- **`json_input`**
    - The JSON input that needs to be converted into a list of strings. It must be a valid JSON string representing a list.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`strings`**
    - Comfy dtype: `STRING`
    - The output list of strings converted from the input JSON list.
    - Python dtype: `List[str]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltJSONListToStringList:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_input": ("STRING", {"forceInput": True}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("strings",)
    OUTPUT_IS_LIST = (True,)

    FUNCTION = "convert"
    CATEGORY = f"{MENU_NAME}/JSON"

    def convert(self, json_input):
        try:
            json_data = json.loads(json_input)
        except json.JSONDecodeError as e:
            raise Exception(f"Error in JSON formatting: {e}")
        
        if not isinstance(json_data, list):
            raise ValueError("Unable to parse `json_input`. Is it a list?")

        return (json_data, )

```
