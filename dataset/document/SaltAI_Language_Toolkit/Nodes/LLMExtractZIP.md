---
tags:
- LLM
---

# ∞ Extract ZIP File
## Documentation
- Class name: `LLMExtractZIP`
- Category: `SALT/IO/Loaders`
- Output node: `False`

The LLMExtractZIP node is designed to extract files from a ZIP archive, with the option to filter the extracted files based on their extensions. It ensures that only desired files are extracted and handles the creation of necessary directories, providing detailed logging of the extraction process.
## Input types
### Required
- **`path`**
    - Defines the destination path where the extracted files will be saved. This parameter is essential for directing the output of the extraction process, ensuring that files are organized and stored in the intended location.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`filter_extensions`**
    - Specifies the file extensions to include in the extraction process, allowing for selective extraction of files. This parameter plays a crucial role in filtering out unwanted files, enhancing the node's efficiency and relevance to the user's needs.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`extraction_path`**
    - Comfy dtype: `STRING`
    - The path to the directory where the files have been extracted. This output is crucial for users to locate and access the extracted files easily.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LLMExtractZIP:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {}),
                "filter_extensions": ("STRING", {"default": "*.*"})
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("extraction_path",)

    CATEGORY = "SALT/IO/Loaders"
    FUNCTION = "extract"

    def extract(self, path: str, filter_extensions: str = "*.*"):
        ext_path = get_full_path(0, str(uuid.uuid4()))
        os.makedirs(ext_path, exist_ok=True)
        
        try:
            with zipfile.ZipFile(path, 'r') as zip_ref:
                all_files = zip_ref.namelist()
                logger.info(f"Files in ZIP: {all_files}")

                if filter_extensions != "*.*":
                    extensions = [ext.strip().lower() for ext in filter_extensions.split(",")]
                    filtered_files = [f for f in all_files if any(f.lower().endswith(ext) for ext in extensions)]
                else:
                    filtered_files = all_files

                logger.info(f"Filtered files: {filtered_files}")

                for file in filtered_files:
                    if not file.endswith('/') and not file.lower().endswith('.ds_store') and not file.startswith('__MACOSX'):
                        normalized_file = file.strip()
                        file_path = os.path.join(ext_path, normalized_file)
                        os.makedirs(os.path.dirname(file_path), exist_ok=True)
                        try:
                            with zip_ref.open(file) as source, open(file_path, "wb") as target:
                                target.write(source.read())
                            logger.info(f"Extracted file: {file}")
                        except FileNotFoundError:
                            errmsg = f"File not found during extraction: {file}"
                            logger.warning(errmsg)
                        except Exception as e:
                            errmsg = f"Failed to extract file `{file}`: {str(e)}"
                            logger.error(errmsg)
        except Exception as e:
            errmsg = f"Failed to extract the zip file `{path}`: {str(e)}"
            logger.error(errmsg)
            raise ValueError(errmsg)
        
        logger.info(f"Zip file {path} extracted to {ext_path}")
        
        return (ext_path,)

```
