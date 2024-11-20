# Validate JSON
## Documentation
- Class name: `SaltValidateJSON`
- Category: `SALT/JSON`
- Output node: `False`

The SaltValidateJSON node is designed to validate JSON input against a specified JSON schema. It checks if the given JSON data adheres to the structure and data types defined in the schema, providing a boolean validity status and an error message if applicable.
## Input types
### Required
- **`json_input`**
    - The JSON string to be validated. It is essential for determining if the input JSON matches the specified schema.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`json_schema`**
    - The JSON schema string against which the input JSON will be validated. This schema defines the expected structure, constraints, and types of the input JSON.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`is_valid`**
    - Comfy dtype: `BOOLEAN`
    - A boolean indicating whether the input JSON is valid according to the specified schema.
    - Python dtype: `bool`
- **`error_message`**
    - Comfy dtype: `STRING`
    - A string containing the error message if the validation fails, otherwise an empty string.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltValidateJSON:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_input": ("STRING", {"forceInput": True}),
                "json_schema": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "STRING")
    RETURN_NAMES = ("is_valid", "error_message")

    FUNCTION = "validate"
    CATEGORY = f"{MENU_NAME}/JSON"

    def validate(self, json_input, json_schema):
        try:
            import jsonschema
        except ImportError:
            raise Exception("jsonschema library is required")

        try:
            json_data = json.loads(json_input)
            schema = json.loads(json_schema)
        except json.JSONDecodeError as e:
            raise Exception(f"Error in JSON formatting: {e}")

        try:
            jsonschema.validate(instance=json_data, schema=schema)
            return (True, "")
        except jsonschema.exceptions.ValidationError as e:
            return (False, str(e))

```
