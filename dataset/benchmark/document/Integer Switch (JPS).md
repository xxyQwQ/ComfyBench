- `Integer Switch (JPS)`: The Integer Switch node is designed to select and output one of several integer inputs based on a specified selection criterion. It facilitates conditional logic within data flows by allowing the dynamic selection of integer values.
    - Inputs:
        - `select` (Required): Determines which integer input to select and output. The selection is based on this integer value, enabling conditional logic and dynamic data flow. Type should be `INT`.
        - `int_i` (Optional): Represents a series of optional integer inputs (int_1 to int_5) that can be selected for output. Each 'int_i' serves as a potential output based on the 'select' criterion. Type should be `INT`.
    - Outputs:
        - `int_out`: The output integer value selected based on the input criteria. Type should be `INT`.