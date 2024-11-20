# ∞ Simple Directory
## Documentation
- Class name: `LLMDirectoryReader`
- Category: `SALT/Language Toolkit/Readers`
- Output node: `False`

The LLMDirectoryReader node specializes in reading and processing directory structures, enabling the extraction and organization of document data from various file types contained within. It serves as a versatile tool for handling bulk document processing tasks, streamlining the workflow for data ingestion from directories.
## Input types
### Required
- **`input_directory`**
    - Specifies the root directory from which documents will be read. This is the starting point for the directory traversal.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
- **`optional_path_list`**
    - A list of specific paths within the directory to include or exclude, providing finer control over which files are processed.
    - Comfy dtype: `*`
    - Python dtype: `List[str] or None`
- **`recursive`**
    - Determines whether the directory reading should be recursive, allowing for the inclusion of files in subdirectories.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`required_ext_list`**
    - A comma-separated list of file extensions to include in the reading process, enabling filtering of files by type.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`exclude_glob_list`**
    - A comma-separated list of patterns to exclude from the reading process, offering a way to skip over unwanted files or directories.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`documents`**
    - Comfy dtype: `DOCUMENT`
    - The output is a collection of documents extracted from the specified directory, organized and ready for further processing.
    - Python dtype: `List[Document]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LLMDirectoryReader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_directory": ("STRING", {}),
            },
            "optional": {
                "optional_path_list": (WILDCARD, {}),
                "recursive": ("BOOLEAN", {"default": False}),
                "required_ext_list": ("STRING", {"default": ".json, .txt, .html"}),
                "exclude_glob_list": ("STRING", {"default": ".sqlite, .zip"}),
            },
        }

    RETURN_TYPES = ("DOCUMENT", )
    RETURN_NAMES = ("documents",)

    FUNCTION = "read_directory"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Readers"

    def read_directory(self, input_directory, optional_path_list=[], recursive=False, required_ext_list=None, exclude_glob_list=None):

        if required_ext_list.strip():
            required_exts = [ext.strip() for ext in required_ext_list.split(",") if ext.strip()]
        else:
            required_exts = None

        if exclude_glob_list.strip():
            exclude = [pattern.strip() for pattern in exclude_glob_list.split(",") if pattern.strip()]
        else:
            exclude = None

        logger.info("Excluding: ", exclude)
        logger.info("Required Extensions: ", required_exts)

        if not optional_path_list:
            full_path = get_full_path(1, input_directory.strip())
            input_dir = full_path if os.path.isdir(full_path) else None
            if not input_dir:
                raise ValueError("The provided subdirectory does not exist.")

            reader = SimpleDirectoryReader(
                input_dir=input_dir,
                exclude_hidden=True,
                recursive=recursive,
                required_exts=required_exts,
                exclude=exclude
            )
        elif optional_path_list and isinstance(optional_path_list, (str, list)):

            if isinstance(optional_path_list, str):
                path_list = [optional_path_list]
            else:
                path_list = []
                for path in optional_path_list:
                    if os.path.isfile(path): # and path.startswith(folder_paths.get_input_directory()):
                        path_list.append(path)

            reader = SimpleDirectoryReader(
                input_files=path_list,
            )


        documents = reader.load_data()
        if not documents:
            raise ValueError("No documents found in the specified directory.")

        return (documents,)

```
