# Get Path Basename
## Documentation
- Class name: `SaltStringPathBasename`
- Category: `SALT/Paths`
- Output node: `False`

This node extracts the basename (the final component of a file path) from a given file path string. It simplifies file path manipulation by isolating the most specific part of the path, such as the file name and extension, from a complete path.
## Input types
### Required
- **`file_path`**
    - The complete file path from which the basename is to be extracted. This input is crucial for determining the specific file or directory name at the end of the path.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`path_basename`**
    - Comfy dtype: `STRING`
    - The basename of the provided file path, which is the file name and extension or the final directory name in the path.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltStringPathBasename:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"forceInput": True})
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path_basename",)

    FUNCTION = "get_basename"
    CATEGORY = f"{MENU_NAME}/Paths"

    def get_basename(self, file_path):
        return(os.path.basename(file_path),)

```
