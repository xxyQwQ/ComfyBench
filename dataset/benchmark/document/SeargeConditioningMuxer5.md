- `SeargeConditioningMuxer5`: The SeargeConditioningMuxer5 node is designed for selecting one of five conditioning inputs based on a specified selector input. It facilitates dynamic control over which conditioning data is passed forward in a processing pipeline, allowing for flexible manipulation of input data streams.
    - Inputs:
        - `input0` (Required): Represents the first conditioning input option. Its selection is dependent on the value of the input selector. Type should be `CONDITIONING`.
        - `input1` (Required): Represents the second conditioning input option, selectable through the input selector. Type should be `CONDITIONING`.
        - `input2` (Required): Denotes the third conditioning input option, which can be chosen based on the input selector's value. Type should be `CONDITIONING`.
        - `input3` (Required): Indicates the fourth conditioning input option, selectable via the input selector. Type should be `CONDITIONING`.
        - `input4` (Required): Signifies the fifth conditioning input option, chosen based on the input selector's value. Type should be `CONDITIONING`.
        - `input_selector` (Required): An integer that selects which of the five conditioning inputs to pass forward. It determines the flow of data through the node. Type should be `INT`.
    - Outputs:
        - `output`: The selected conditioning input, determined by the input selector. It enables dynamic data flow control within the pipeline. Type should be `CONDITIONING`.
