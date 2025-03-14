- `ImpactRemoteBoolean`: The ImpactRemoteBoolean node is designed to interact with boolean widgets in a remote interface, allowing for the dynamic control of boolean values based on external inputs.
    - Inputs:
        - `node_id` (Required): Specifies the unique identifier of the node whose widget is being controlled, playing a crucial role in targeting the correct widget for value updates. Type should be `INT`.
        - `widget_name` (Required): Identifies the specific widget within the node to be controlled, enabling precise manipulation of its boolean value. Type should be `STRING`.
        - `value` (Required): The boolean value to be set for the specified widget, dictating the widget's state as either true or false. Type should be `BOOLEAN`.
    - Outputs:
