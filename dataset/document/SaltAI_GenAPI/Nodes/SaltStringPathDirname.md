# Get Path Directory
## Documentation
- Class name: `SaltStringPathDirname`
- Category: `SALT/Paths`
- Output node: `False`

This node extracts the directory path from a given file path, effectively isolating the portion of the path that specifies the directory containing the file. It serves to simplify file path manipulation by providing a straightforward method to obtain the directory component.
## Input types
### Required
- **`file_path`**
    - The file path from which the directory name is to be extracted. This input is crucial for determining the specific directory location within a filesystem.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`path_directory`**
    - Comfy dtype: `STRING`
    - The directory portion of the provided file path, which specifies the location of the file within the filesystem.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltStringPathDirname:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"forceInput": True})
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path_directory",)

    FUNCTION = "get_basename"
    CATEGORY = f"{MENU_NAME}/Paths"

    def get_basename(self, file_path):
        return(os.path.dirname(file_path),)

```
