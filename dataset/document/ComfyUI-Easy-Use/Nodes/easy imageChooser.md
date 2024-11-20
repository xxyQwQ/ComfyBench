---
tags:
- Image
---

# Image Chooser
## Documentation
- Class name: `easy imageChooser`
- Category: `EasyUse/Image`
- Output node: `True`

The `easy imageChooser` node enables users to select preferred images from a given set through an interactive interface. This node simplifies the process of image selection, making it an essential tool for workflows that require user engagement and decision-making in choosing specific images for further processing or analysis.
## Input types
### Required
- **`mode`**
    - Specifies the selection mode, allowing users to define how the image selection process should be conducted. This parameter determines the behavior of the chooser, affecting the user's interaction and the selection criteria.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
### Optional
- **`images`**
    - A collection of images presented to the user for selection. This input is essential for providing the set of options from which the user can make their choice, directly influencing the selection process.
    - Comfy dtype: `IMAGE`
    - Python dtype: `List[torch.Tensor]`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - Returns the selected image or images, reflecting the user's choices. This output is vital for further processing or analysis of the chosen images.
    - Python dtype: `List[torch.Tensor]`
- **`ui`**
    - Provides the user interface elements necessary for displaying the images to the user and capturing their selections. It is crucial for the interactive aspect of the node, enabling users to visually select their preferred images.
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class imageChooser(PreviewImage):
  @classmethod
  def INPUT_TYPES(self):
    return {
      "required":{
        "mode": (['Always Pause', 'Keep Last Selection'], {"default": "Always Pause"}),
      },
      "optional": {
        "images": ("IMAGE",),
      },
      "hidden": {"prompt": "PROMPT", "my_unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO"},
    }

  RETURN_TYPES = ("IMAGE",)
  RETURN_NAMES = ("image",)
  FUNCTION = "chooser"
  OUTPUT_NODE = True
  INPUT_IS_LIST = True
  CATEGORY = "EasyUse/Image"

  last_ic = {}
  @classmethod
  def IS_CHANGED(cls, my_unique_id, **kwargs):
    return cls.last_ic[my_unique_id[0]]

  def tensor_bundle(self, tensor_in: torch.Tensor, picks):
    if tensor_in is not None and len(picks):
      batch = tensor_in.shape[0]
      return torch.cat(tuple([tensor_in[(x) % batch].unsqueeze_(0) for x in picks])).reshape(
        [-1] + list(tensor_in.shape[1:]))
    else:
      return None

  def chooser(self, prompt=None, my_unique_id=None, extra_pnginfo=None, **kwargs):
    id = my_unique_id[0]
    if id not in ChooserMessage.stash:
      ChooserMessage.stash[id] = {}
    my_stash = ChooserMessage.stash[id]

    # enable stashing. If images is None, we are operating in read-from-stash mode
    if 'images' in kwargs:
      my_stash['images'] = kwargs['images']
    else:
      kwargs['images'] = my_stash.get('images', None)

    if (kwargs['images'] is None):
      return (None, None, None, "")

    images_in = torch.cat(kwargs.pop('images'))
    self.batch = images_in.shape[0]
    for x in kwargs: kwargs[x] = kwargs[x][0]
    result = self.save_images(images=images_in, prompt=prompt)

    images = result['ui']['images']
    PromptServer.instance.send_sync("easyuse-image-choose", {"id": id, "urls": images})

    # 获取上次选择
    mode = kwargs.pop('mode', 'Always Pause')
    last_choosen = None
    if mode == 'Keep Last Selection':
      if not extra_pnginfo:
        print("Error: extra_pnginfo is empty")
      elif (not isinstance(extra_pnginfo[0], dict) or "workflow" not in extra_pnginfo[0]):
        print("Error: extra_pnginfo[0] is not a dict or missing 'workflow' key")
      else:
        workflow = extra_pnginfo[0]["workflow"]
        node = next((x for x in workflow["nodes"] if str(x["id"]) == id), None)
        if node:
          last_choosen = node['properties']['values']

    # wait for selection
    try:
      selections = ChooserMessage.waitForMessage(id, asList=True) if last_choosen is None or len(last_choosen)<1 else last_choosen
      choosen = [x for x in selections if x >= 0] if len(selections)>1 else [0]
    except ChooserCancelled:
      raise comfy.model_management.InterruptProcessingException()

    return {"ui": {"images": images},
              "result": (self.tensor_bundle(images_in, choosen),)}

```
