- `ImpactInversedSwitch`: This node is designed to selectively invert the input signal based on a specified condition. It operates by examining the 'select' input and, depending on its value and type, either directly uses the input value or retrieves and uses a specific value from another node. This functionality allows for dynamic control flow and decision-making within a node network, enabling the inversion of signals based on runtime conditions.
    - Inputs:
        - `select` (Required): The 'select' input determines the condition under which the input signal is inverted. It can be a direct value or a reference to another node's output, allowing for dynamic and conditional inversion based on the network's state. Type should be `INT`.
        - `input` (Required): The 'input' parameter represents the signal to be potentially inverted. The inversion is conditional, based on the evaluation of the 'select' parameter. Type should be `*`.
    - Outputs:
        - `*`: unknown Type should be `*`.