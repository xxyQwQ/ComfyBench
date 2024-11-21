- `ImpactMinMax`: The ImpactMinMax node is designed to compare two input values and return either the maximum or minimum value based on a specified mode. This functionality is essential for operations requiring conditional selection between two values, such as optimizing performance or making decisions based on dynamic input.
    - Inputs:
        - `mode` (Required): Determines whether the maximum or minimum of the two inputs will be returned. When true, the maximum value is selected; otherwise, the minimum value is chosen. Type should be `BOOLEAN`.
        - `a` (Required): One of the two values to be compared. This input, along with 'b', is essential for determining the output based on the selected mode. Type should be `*`.
        - `b` (Required): The second of the two values to be compared. This input is crucial for the comparison operation alongside 'a'. Type should be `*`.
    - Outputs:
        - `int`: The result of the comparison, either the maximum or minimum value between the two inputs, depending on the mode. Type should be `INT`.