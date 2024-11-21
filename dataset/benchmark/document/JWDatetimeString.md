- `JWDatetimeString`: This node generates a string representation of the current datetime, formatted according to a specified pattern. It abstracts the complexity of datetime formatting, providing a simple interface for obtaining formatted datetime strings.
    - Inputs:
        - `format` (Required): Specifies the format in which the current datetime should be returned. This allows for customization of the output string according to the needs of the application. Type should be `STRING`.
    - Outputs:
        - `string`: The output is a string that represents the current datetime, formatted according to the specified pattern. Type should be `STRING`.