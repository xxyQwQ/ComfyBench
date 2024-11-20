# Load Text File by Filename/Index
## Documentation
- Class name: `SaltLoadFileByIndexOrName`
- Category: `SALT/IO`
- Output node: `False`

This node is designed to load a file's content based on either its index within a directory or its name. It streamlines the process of retrieving file data, making it accessible for further processing or analysis.
## Input types
### Required
- **`path`**
    - Specifies the path to the directory containing the file to be loaded. It is crucial for locating the target file within the filesystem.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`file_identifier`**
    - Acts as a unique identifier for the file to be loaded, which can be either the file's name or its index within the specified directory. This flexibility allows for easy reference to files without needing to know their exact names.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`file_content`**
    - Comfy dtype: `STRING`
    - The content of the loaded file, returned as a string. This output makes the file's data readily available for use in subsequent operations.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltLoadFileByIndexOrName:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"forceInput": True}),
                "file_identifier": ("STRING", {"default": "0"})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_content",)

    FUNCTION = "load_file"
    CATEGORY = f"{MENU_NAME}/IO"

    def load_file(self, directory_path, file_identifier):
        input_dir = folder_paths.get_input_directory()
        temp_dir = folder_paths.get_temp_directory()
        abs_directory_path = get_abs_path(directory_path, input_dir, temp_dir)

        if not os.path.isdir(abs_directory_path):
            msg = f"Provided path is not a directory: {directory_path}"
            logger.error(msg)
            raise Exception(msg)

        files = [f for f in os.listdir(abs_directory_path) if is_valid_file(f)]
        files.sort()

        try:
            if file_identifier.isdigit():
                file_index = int(file_identifier)
                if file_index >= len(files):
                    raise IndexError("File index out of range")
                file_name = files[file_index]
            else:
                if file_identifier not in files:
                    raise FileNotFoundError(f"File not found: {file_identifier}")
                file_name = file_identifier

            file_path = os.path.join(abs_directory_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                file_content = file.read()

            return (file_content,)
        
        except (IndexError, FileNotFoundError) as e:
            msg = str(e)
            logger.error(msg)
            raise Exception(msg)

```
