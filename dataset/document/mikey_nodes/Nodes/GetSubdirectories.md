# Get Subdirectories (Mikey)
## Documentation
- Class name: `GetSubdirectories`
- Category: `Mikey/Utils`
- Output node: `False`

This node is designed to retrieve all subdirectories within a specified directory, effectively scanning a directory path to list its immediate child directories. It serves as a utility for directory management and organization, facilitating the exploration and manipulation of filesystem structures.
## Input types
### Required
- **`directory`**
    - Specifies the directory path to scan for subdirectories. It is the starting point for the search operation, determining where the node begins its directory traversal.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`subdirectories`**
    - Comfy dtype: `STRING`
    - Returns a list of paths representing the subdirectories found within the specified directory. This output enables further processing or inspection of these directories.
    - Python dtype: `List[str]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class GetSubdirectories:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'directory': ('STRING', {'default': 'base_directory'})}}

    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('subdirectories',)
    FUNCTION = 'get_subdirectories'
    OUTPUT_IS_LIST = (True, )
    CATEGORY = 'Mikey/Utils'

    def get_subdirectories(self, directory):
        if os.path.isdir(directory):
            subdirectories = [f.path for f in os.scandir(directory) if f.is_dir()]
        else:
            raise Exception(f'{directory} is not a valid directory')
        return (subdirectories,)

```
