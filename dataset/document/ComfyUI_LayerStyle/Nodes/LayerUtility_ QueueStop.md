# LayerUtility: Queue Stop
## Documentation
- Class name: `LayerUtility: QueueStop`
- Category: `😺dzNodes/LayerUtility/SystemIO`
- Output node: `False`

This node is designed to control the execution flow within a queue by providing the functionality to either stop or continue the queue based on specified conditions. It plays a crucial role in managing the execution order and conditions within complex workflows, allowing for dynamic control over the process flow.
## Input types
### Required
- **`any`**
    - A generic input parameter that can be of any type, serving as a placeholder for data that might be passed through the node without alteration.
    - Comfy dtype: `*`
    - Python dtype: `Any`
- **`mode`**
    - Specifies the operation mode of the node, which can either be 'stop' to halt the queue or 'continue' to proceed without interruption. This allows for conditional control over the queue's execution.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
- **`stop`**
    - A boolean parameter that, when true and the mode is set to 'stop', triggers the stopping of the queue. This provides a direct mechanism to halt the queue's execution based on logical conditions.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`any`**
    - Comfy dtype: `*`
    - Returns the input data unmodified, allowing for the data to be passed through or halted based on the node's execution logic.
    - Python dtype: `Any`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class QueueStopNode():
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        mode_list = ["stop", "continue"]
        return {
            "required": {
                "any": (any, ),
                "mode": (mode_list,),
                "stop": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = (any,)
    RETURN_NAMES = ("any",)
    FUNCTION = 'stop_node'
    CATEGORY = '😺dzNodes/LayerUtility/SystemIO'

    def stop_node(self, any, mode,stop):
        if mode == "stop":
            if stop:
                log(f"Queue stopped, it was terminated by node.", "error")
                from comfy.model_management import InterruptProcessingException
                raise InterruptProcessingException()

        return (any,)

```
