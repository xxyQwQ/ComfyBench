# Text File Line Iterator
## Documentation
- Class name: `TextFileLineIterator`
- Category: `cspnodes`
- Output node: `False`

The TextFileLineIterator node is designed to read and return a specific line from a text file, based on a given line index. It abstracts the process of file handling, reading, and line selection, making it easier to extract text data programmatically.
## Input types
### Required
- **`file_path`**
    - Specifies the path to the text file from which a line will be read. It is crucial for locating and accessing the file's content.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`line_index`**
    - Determines the index of the line to be retrieved from the text file. The index is wrapped around using modulo to ensure it falls within the file's line count, allowing cyclic access.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`string`**
    - Comfy dtype: `STRING`
    - Returns the text of the specified line from the text file, stripped of any surrounding whitespace.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class TextFileLineIterator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {}),
                "line_index": ("INT", {"default": 0})
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "get_line_by_index"
    CATEGORY = "cspnodes"

    def get_line_by_index(self, file_path, line_index):
        # Read all lines from the text file
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        # Wrap the index around using modulo
        line_index = line_index % len(lines)

        # Get the specified line and strip any surrounding whitespace
        line = lines[line_index].strip()

        return (line,)

```
