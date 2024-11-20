---
tags:
- ConditionalSelection
- ImpactPack
---

# Control Bridge
## Documentation
- Class name: `ImpactControlBridge`
- Category: `ImpactPack/Logic/_for_test`
- Output node: `True`

The ImpactControlBridge node serves as a dynamic control mechanism for managing the state of other nodes within a network based on specific conditions. It can activate, mute, or bypass nodes based on the operational mode, thereby altering the flow of data or processing within the system.
## Input types
### Required
- **`value`**
    - The input value that is passed through the node unchanged if none of the conditions for activation, muting, or bypassing are met. This value can be of any type, reflecting the diverse data that may flow through the system.
    - Comfy dtype: `*`
    - Python dtype: `object`
- **`mode`**
    - Determines if the node should activate other nodes. Activation is based on the presence of nodes in the 'should_be_active_nodes' list, which is a combination of nodes specified to be muted and bypassed.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`behavior`**
    - Controls the muting of nodes. When set, nodes in the 'should_be_mute_nodes' list, which includes active and bypass nodes, are muted to prevent them from processing data.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`value`**
    - Comfy dtype: `*`
    - The output value, which is the input value passed through unchanged unless the node's processing is interrupted by activating, muting, or bypassing other nodes.
    - Python dtype: `object`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ImpactControlBridge:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                      "value": (any_typ,),
                      "mode": ("BOOLEAN", {"default": True, "label_on": "Active", "label_off": "Mute/Bypass"}),
                      "behavior": ("BOOLEAN", {"default": True, "label_on": "Mute", "label_off": "Bypass"}),
                    },
                "hidden": {"unique_id": "UNIQUE_ID", "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"}
                }

    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Logic/_for_test"
    RETURN_TYPES = (any_typ,)
    RETURN_NAMES = ("value",)
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(self, value, mode, behavior=True, unique_id=None, prompt=None, extra_pnginfo=None):
        # NOTE: extra_pnginfo is not populated for IS_CHANGED.
        #       so extra_pnginfo is useless in here
        try:
            workflow = core.current_prompt['extra_data']['extra_pnginfo']['workflow']
        except:
            print(f"[Impact Pack] core.current_prompt['extra_data']['extra_pnginfo']['workflow']")
            return 0

        nodes, links = workflow_to_map(workflow)
        next_nodes = []

        for link in nodes[unique_id]['outputs'][0]['links']:
            node_id = str(links[link][2])
            impact.utils.collect_non_reroute_nodes(nodes, links, next_nodes, node_id)

        return next_nodes

    def doit(self, value, mode, behavior=True, unique_id=None, prompt=None, extra_pnginfo=None):
        global error_skip_flag

        workflow_nodes, links = workflow_to_map(extra_pnginfo['workflow'])

        active_nodes = []
        mute_nodes = []
        bypass_nodes = []

        for link in workflow_nodes[unique_id]['outputs'][0]['links']:
            node_id = str(links[link][2])

            next_nodes = []
            impact.utils.collect_non_reroute_nodes(workflow_nodes, links, next_nodes, node_id)

            for next_node_id in next_nodes:
                node_mode = workflow_nodes[next_node_id]['mode']

                if node_mode == 0:
                    active_nodes.append(next_node_id)
                elif node_mode == 2:
                    mute_nodes.append(next_node_id)
                elif node_mode == 4:
                    bypass_nodes.append(next_node_id)

        if mode:
            # active
            should_be_active_nodes = mute_nodes + bypass_nodes
            if len(should_be_active_nodes) > 0:
                PromptServer.instance.send_sync("impact-bridge-continue", {"node_id": unique_id, 'actives': list(should_be_active_nodes)})
                nodes.interrupt_processing()

        elif behavior:
            # mute
            should_be_mute_nodes = active_nodes + bypass_nodes
            if len(should_be_mute_nodes) > 0:
                PromptServer.instance.send_sync("impact-bridge-continue", {"node_id": unique_id, 'mutes': list(should_be_mute_nodes)})
                nodes.interrupt_processing()

        else:
            # bypass
            should_be_bypass_nodes = active_nodes + mute_nodes
            if len(should_be_bypass_nodes) > 0:
                PromptServer.instance.send_sync("impact-bridge-continue", {"node_id": unique_id, 'bypasses': list(should_be_bypass_nodes)})
                nodes.interrupt_processing()

        return (value, )

```
