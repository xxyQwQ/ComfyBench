# GetCategoryByIndex
## Documentation
- Class name: `GetCategoryByIndex`
- Category: `SALT/CLUSTERING`
- Output node: `False`

The GetCategoryByIndex node is designed to filter and retrieve specific categories of data based on a given index. It operates by selecting subsets of features and captions that match the specified index, effectively isolating a particular category from a larger dataset.
## Input types
### Required
- **`id`**
    - The 'id' parameter specifies the index of the category to be retrieved. It determines which category's features and captions are selected, playing a crucial role in the node's operation by filtering the dataset according to this index.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`features`**
    - The 'features' parameter represents the numerical data or attributes associated with each item in the dataset. It is essential for identifying and isolating the category corresponding to the given index.
    - Comfy dtype: `NP_ARRAY`
    - Python dtype: `numpy.ndarray`
- **`captions`**
    - The 'captions' parameter consists of textual descriptions or titles associated with each item in the dataset. It is used alongside 'id' and 'features' to select and return the relevant captions for the specified category.
    - Comfy dtype: `LIST`
    - Python dtype: `list`
- **`labels`**
    - The 'labels' parameter contains the indices or identifiers for each category within the dataset. It is used to match the 'id' parameter and filter the dataset to retrieve the specified category's data.
    - Comfy dtype: `LIST`
    - Python dtype: `list`
## Output types
- **`features`**
    - Comfy dtype: `NP_ARRAY`
    - Returns the filtered features corresponding to the specified category index, isolating a subset of the dataset's numerical attributes.
    - Python dtype: `numpy.ndarray`
- **`captions`**
    - Comfy dtype: `LIST`
    - Returns the filtered captions corresponding to the specified category index, providing textual descriptions for the isolated subset of items.
    - Python dtype: `list`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class GetCategoryByIndex:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "id": ("INT", {"default": 0, "min": -1}),
                "features": ("NP_ARRAY", ),
                "captions": ("LIST", ),
                "labels": ("LIST", ),
            }
        }
    RETURN_TYPES = ("NP_ARRAY", "LIST")
    RETURN_NAMES = ("features", "captions")

    FUNCTION = "get_category_by_id"
    CATEGORY = f"{MENU_NAME}/CLUSTERING"

    def get_category_by_id(self, id, features, captions, labels):
        try:
            captions = [captions[idx] for idx in range(len(captions)) if labels[idx] == id]
            features = features[labels == id]
        except:
            return (np.array([]), [])
        return (features, captions)

```
