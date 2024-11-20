# Load Text
## Documentation
- Class name: `Load Text`
- Category: `Hakkun`
- Output node: `False`

The Load Text node is designed to read and process text data from a specified file path, filtering out any lines that start with a comment symbol. It abstracts the file reading and preprocessing steps, providing a clean text output for further use.
## Input types
### Required
- **`file_path`**
    - Specifies the path to the text file to be loaded. This parameter is essential for locating and reading the file's contents.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`string`**
    - Comfy dtype: `STRING`
    - Returns the processed text from the file, with comments removed and lines cleaned up.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LoadText:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"default": '', "multiline": False}),
            }
        }

    RETURN_TYPES = (TEXT_TYPE,)
    FUNCTION = "load_file"

    CATEGORY = "Hakkun"

    def load_file(self, file_path=''):
        return (load_text(file_path),)

```
