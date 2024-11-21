- `ImpactNodeSetMuteState`: This node is designed to control the mute state of other nodes within a workflow. It sends a command to mute or activate specified nodes based on the provided state, enhancing the flexibility and control over the workflow's execution.
    - Inputs:
        - `signal` (Required): A generic signal input that triggers the node's operation. It's essential for initiating the mute or activation process. Type should be `*`.
        - `node_id` (Required): Specifies the unique identifier of the node whose mute state is to be controlled. It determines the target node for the mute or activation command. Type should be `INT`.
        - `set_state` (Required): A boolean value indicating the desired mute state. When true, the target node is activated; when false, it is muted. Type should be `BOOLEAN`.
    - Outputs:
        - `signal_opt`: Returns the original signal input, allowing for seamless integration into the workflow without altering the data flow. Type should be `*`.