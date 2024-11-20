# JSON Compile Files from Directory
## Documentation
- Class name: `SaltLoadDirectoryAsJSON`
- Category: `SALT/IO`
- Output node: `False`

This node is designed to compile files from a specified directory into a single JSON object. It reads each file within the directory, assuming they are in JSON format, and combines them into a unified JSON structure. The method of combination can vary based on the mode specified, such as using filenames as keys, which allows for a structured and accessible way to handle multiple JSON files collectively.
## Input types
### Required
- **`path`**
    - Specifies the path to the directory containing the JSON files to be compiled. This path is crucial as it determines the source of the JSON files that will be processed and combined.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`mode`**
    - Determines the method used to compile the JSON files into a single JSON object. For example, using filenames as keys to associate each file's content with its name in the resulting JSON structure.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`reverse_order`**
    - A boolean flag indicating whether the files should be processed in reverse order. This can affect the final structure of the compiled JSON, especially if the order of files is significant.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`json_output`**
    - Comfy dtype: `STRING`
    - The resulting JSON object after compiling the files from the specified directory. It represents a unified structure containing the contents of all processed files.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltLoadDirectoryAsJSON:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"forceInput": True}),
                "mode": (["simple_list", "update_dict", "filename_as_key"],),
                "reverse_order": ("BOOLEAN", {"default": False})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json_output",)

    FUNCTION = "load_directory"
    CATEGORY = f"{MENU_NAME}/IO"

    def load_directory(self, directory_path, mode, reverse_order):
        input_dir = folder_paths.get_input_directory()
        temp_dir = folder_paths.get_temp_directory()
        abs_directory_path = get_abs_path(directory_path.strip(), input_dir, temp_dir)

        if not os.path.isdir(abs_directory_path):
            msg = f"Provided path is not a directory: {directory_path}"
            logger.error(msg)
            raise Exception(msg)

        json_files = [f for f in os.listdir(abs_directory_path) if f.endswith('.json')]
        json_files.sort(reverse=True if reverse_order else False)

        if mode == "simple_list":
            json_list = []
            for file in json_files:
                file_path = os.path.join(abs_directory_path, file)
                with open(file_path, 'r') as f:
                    try:
                        json_data = json.load(f)
                        json_list.append(json_data)
                    except json.JSONDecodeError as e:
                        raise Exception(f"Error in JSON formatting in file {file}: {e}")
            return (json.dumps(json_list),)

        elif mode == "update_dict":
            combined_json = {}
            for file in json_files:
                file_path = os.path.join(abs_directory_path, file)
                with open(file_path, 'r') as f:
                    try:
                        json_data = json.load(f)
                        combined_json.update(json_data)
                    except json.JSONDecodeError as e:
                        raise Exception(f"Error in JSON formatting in file {file}: {e}")
            return (json.dumps(combined_json),)

        elif mode == "filename_as_key":
            combined_json = {}
            for file in json_files:
                file_path = os.path.join(abs_directory_path, file)
                with open(file_path, 'r') as f:
                    try:
                        json_data = json.load(f)
                        file_name_without_ext = os.path.splitext(file)[0]
                        combined_json[file_name_without_ext] = json_data
                    except json.JSONDecodeError as e:
                        raise Exception(f"Error in JSON formatting in file {file}: {e}")
            return (json.dumps(combined_json),)

```
