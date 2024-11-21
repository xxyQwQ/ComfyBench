- `DF_To_text_(Debug)`: The `DF_To_text_(Debug)` node is designed for debugging purposes, allowing users to log and inspect any data passed through it. It converts the input data to a string format, logs the original and string-converted data for debugging, and handles exceptions by logging them. This node facilitates the observation and troubleshooting of data flow within node-based processing pipelines.
    - Inputs:
        - `ANY` (Required): Accepts any type of data for debugging purposes. It logs the input data in its original form and after conversion to a string, aiding in the inspection and troubleshooting of data flow. Type should be `*`.
    - Outputs:
        - `SAME AS INPUT`: Returns the original input data, allowing it to be passed through for further processing or inspection. Type should be `*`.
        - `STRING`: Returns the input data converted to a string, or an error message if an exception occurs, facilitating debugging and error tracking. Type should be `STRING`.