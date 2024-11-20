# Remove Local Image
## Documentation
- Class name: `easy removeLocalImage`
- Category: `EasyUse/Image`
- Output node: `True`

The `easy removeLocalImage` node is designed to search for and delete a specified local image file. It operates by traversing directories to find a file that matches the given name, removing it if found, and providing feedback on the operation's success or failure.
## Input types
### Required
- **`any`**
    - This parameter is used internally for ensuring the node's compatibility with the system's requirements, without affecting the node's core functionality of searching and deleting files.
    - Comfy dtype: `*`
    - Python dtype: `str`
- **`file_name`**
    - Specifies the name of the file to be searched and deleted. This parameter is crucial for identifying the target file within the local storage.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
The node doesn't have output types
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class removeLocalImage:

  def __init__(self):
    self.hasFile = False

  @classmethod
  def INPUT_TYPES(s):
      return {
        "required": {
          "any": (AlwaysEqualProxy("*"),),
          "file_name": ("STRING",{"default":""}),
        },
      }

  RETURN_TYPES = ()
  OUTPUT_NODE = True
  FUNCTION = "remove"
  CATEGORY = "EasyUse/Image"



  def remove(self, any, file_name):
    self.hasFile = False
    def listdir(path, dir_name=''):
      for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
          dir_name = os.path.basename(file_path)
          listdir(file_path, dir_name)
        else:
          file = os.path.join(dir_name, file)
          name_without_extension, file_extension = os.path.splitext(file)
          if name_without_extension == file_name or file == file_name:
            os.remove(os.path.join(folder_paths.input_directory, file))
            self.hasFile = True
            break

    listdir(folder_paths.input_directory, '')

    if self.hasFile:
      PromptServer.instance.send_sync("easyuse-toast", {"content": "Removed SuccessFully", "type":'success'})
    else:
      PromptServer.instance.send_sync("easyuse-toast", {"content": "Removed Failed", "type": 'error'})
    return ()

```
