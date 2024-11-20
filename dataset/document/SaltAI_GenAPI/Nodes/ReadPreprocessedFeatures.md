# ReadPreprocessedFeatures
## Documentation
- Class name: `ReadPreprocessedFeatures`
- Category: `SALT/CLUSTERING`
- Output node: `False`

This node is designed to read preprocessed feature data from a specified file path, loading it into a numpy array format for further processing or analysis.
## Input types
### Required
- **`file_path`**
    - Specifies the path to the file containing the preprocessed features. This path is essential for locating and loading the data correctly.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`features`**
    - Comfy dtype: `NP_ARRAY`
    - Outputs the preprocessed features loaded from the file as a numpy array, ready for use in subsequent processing steps.
    - Python dtype: `numpy.ndarray`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ReadPreprocessedFeatures:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "file_path": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("NP_ARRAY",)
    RETURN_NAMES = ("features",)

    FUNCTION = "read_features"
    CATEGORY = f"{MENU_NAME}/CLUSTERING"

    def read_features(self, file_path):
        np_array = np.loadtxt(file_path, delimiter=",")
        return (np_array,)

```
