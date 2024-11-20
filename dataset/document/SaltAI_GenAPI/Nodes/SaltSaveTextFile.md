# Save Text File
## Documentation
- Class name: `SaltSaveTextFile`
- Category: `SALT/IO`
- Output node: `False`

This node is designed to save textual content to a file or multiple files within a newly created directory. It supports specifying file names, extensions, and content, handling both single and multiple content inputs. The node ensures the creation of a unique directory for file storage, validating file extensions against supported types, and provides error handling for file creation and saving processes.
## Input types
### Required
- **`filename_prefix`**
    - Specifies the prefix for the file name(s) to be saved. This prefix is used as the base name for the file or files, with indexes appended for multiple contents.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`file_extension`**
    - Defines the file extension for the saved file(s). This extension indicates the file type and is validated against a list of supported types to ensure compatibility.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
### Optional
- **`file_content`**
    - The content to be saved into a single file. This parameter allows for saving a specific piece of text content directly to a file.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`file_contents_input`**
    - A list of text contents to be saved into multiple files. Each content in the list is saved to a separate file, with filenames generated based on the filename prefix and index.
    - Comfy dtype: `LIST`
    - Python dtype: `List[str]`
## Output types
- **`file_paths`**
    - Comfy dtype: `LIST`
    - A list of paths to the saved files. This includes the complete path for each file created during the node's execution.
    - Python dtype: `List[str]`
- **`directory_path`**
    - Comfy dtype: `STRING`
    - The path to the directory where all files are saved. This directory is uniquely generated for each execution of the node.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltSaveTextFile:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filename_prefix": ("STRING", {"default": "file_"}),
                "file_extension": (list(VALID_FILE_TYPES.keys()),),
            },
            "optional": {
                "file_content": ("STRING", {"forceInput":True}),
                "file_contents_input": ("LIST",),
            }
        }

    RETURN_TYPES = ("LIST", "STRING")
    RETURN_NAMES = ("file_paths", "directory_path")

    FUNCTION = "save"
    CATEGORY = f"{MENU_NAME}/IO"

    def save(self, filename_prefix, file_extension, file_content=None, file_contents_input=None):
        input_dir = folder_paths.get_input_directory()
        if file_content is None and file_contents_input is None:
            msg = "Either 'file_content' or 'file_contents' must be provided."
            logger.error(msg)
            raise Exception(msg)

        if not self.valid_ext(file_extension):
            msg = f"Unsupported filetype: {file_extension}"
            logger.error(msg)
            raise Exception(msg)

        dir_uuid = str(uuid.uuid4())
        files_path = os.path.join(os.path.abspath(input_dir), dir_uuid) 

        try:
            os.makedirs(files_path, exist_ok=True)
        except Exception as e:
            msg = f"Failed to create directory: {e}"
            logger.error(msg)
            raise Exception(msg)

        file_paths = []
        if file_content:
            file_path = self.save_file(files_path, f"{filename_prefix}.{file_extension}", file_content)
            file_paths.append(file_path)

        if file_contents_input:
            for idx, content in enumerate(file_contents_input):
                file_path = self.save_file(files_path, f"{filename_prefix}{idx}.{file_extension}", content)
                file_paths.append(file_path)

        return (file_paths, files_path)

    def save_file(self, directory, filename, content):
        file_path = os.path.join(directory, filename)
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(content)
        except Exception as e:
            msg = f"Error saving file {filename}: {e}"
            logger.error(msg)
            raise Exception(msg)
        return file_path

    def valid_ext(self, extension):
        if extension.lower() in list(VALID_FILE_TYPES.keys()):
            return True
        return False

```
