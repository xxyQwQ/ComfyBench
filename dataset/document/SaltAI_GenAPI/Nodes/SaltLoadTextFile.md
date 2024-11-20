---
tags:
- Audio
- SaltNodes
---

# Load Text File
## Documentation
- Class name: `SaltLoadTextFile`
- Category: `SALT/IO`
- Output node: `False`

This node is designed to load the contents of a text file into the system. It focuses on reading and importing text data from a specified file path, making it accessible for further processing or analysis within the application.
## Input types
### Required
- **`file_path`**
    - Specifies the file path of the text file to be loaded. This path is essential for locating and accessing the file for reading.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`file_content`**
    - Comfy dtype: `STRING`
    - The content of the loaded text file. This output makes the text data available for use within the application.
    - Python dtype: `str`
- **`file_log`**
    - Comfy dtype: `STRING`
    - Provides a log message related to the file loading process, including information on mime type determination or errors encountered.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltLoadTextFile:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("file_content", "file_log")

    FUNCTION = "load"
    CATEGORY = f"{MENU_NAME}/IO"

    def load(self, file_path):
        input_dir = folder_paths.get_input_directory()
        temp_dir = folder_paths.get_temp_directory()
        abs_file_path = get_abs_path(file_path.strip(), input_dir, temp_dir)

        mime_type, _ = mimetypes.guess_type(abs_file_path)
        if not mime_type:
            mime_type = guess_mime_type_fallback(abs_file_path)
            logger.warning(f"Unable to determine mime_type, using fallback. Determined mime type to be `{mime_type}` based on extension.")

        if mime_type not in list(VALID_FILE_TYPES.values()):
            msg = f"Unsupported file type: {mime_type}"
            logger.error(msg)
            raise Exception(msg)

        try:
            with open(abs_file_path, 'r', encoding='utf-8') as file:
                file_content = file.read()
        except FileNotFoundError:
            raise Exception(f"File not found: {abs_file_path}")
        except Exception as e:
            raise Exception(f"Error reading file: {e}")

        file_msg = self.validate_file_content(file_content, mime_type)

        return (file_content, file_msg)

    def validate_file_content(self, content, mime_type):
        if mime_type in VALID_FILE_TYPES['json']:
            file_msg = self.validate_json(content)
        elif mime_type in VALID_FILE_TYPES['yaml']:
            file_msg = self.validate_yaml(content)
        elif mime_type in VALID_FILE_TYPES['xml']:
            file_msg = self.validate_xml(content)
        elif mime_type in VALID_FILE_TYPES['csv']:
            file_msg = self.validate_csv(content)
        else:
            file_msg = "Text file. No validation tests performed."
        return file_msg

    def validate_json(self, content):
        try:
            json.loads(content)
        except json.JSONDecodeError as e:
            msg = f"JSON validation error: {e}"
            logger.warning(msg)
            return msg
        return "Valid JSON"

    def validate_yaml(self, content):
        try:
            yaml.safe_load(content)
        except yaml.YAMLError as e:
            msg = f"YAML validation error: {e}"
            logger.warning(msg)
            return msg
        return "Valid YAML"

    def validate_xml(self, content):
        try:
            ET.fromstring(content)
        except ET.ParseError as e:
            msg = f"XML validation error: {e}"
            logger.warning(msg)
            return msg
        return "Valid XML"

    def validate_csv(self, content):
        try:
            csv.reader(StringIO(content))
        except csv.Error as e:
            msg = f"CSV validation error: {e}"
            logger.warning(msg)
            return msg
        return "Valid CSV"

```
