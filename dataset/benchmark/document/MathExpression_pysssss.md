- `MathExpression_pysssss`: The MathExpression node evaluates mathematical expressions dynamically, supporting basic arithmetic, comparisons, and logical operations. It can handle variables and function calls within the expression, offering a flexible way to compute results based on input parameters.
    - Inputs:
        - `expression` (Required): The mathematical expression to be evaluated. Supports arithmetic, comparisons, logical operations, and function calls, with the ability to include variables 'a', 'b', and 'c'. Type should be `STRING`.
        - `a` (Optional): An optional variable that can be used within the expression. Supports integers and floats. Type should be `INT,FLOAT,IMAGE,LATENT`.
        - `b` (Optional): An optional variable that can be used within the expression. Supports integers and floats. Type should be `INT,FLOAT,IMAGE,LATENT`.
        - `c` (Optional): An optional variable that can be used within the expression. Supports integers and floats. Type should be `INT,FLOAT,IMAGE,LATENT`.
    - Outputs:
        - `int`: The integer part of the evaluated result, representing the outcome of the mathematical expression as an integer. Type should be `INT`.
        - `float`: The floating-point part of the evaluated result, representing the outcome of the mathematical expression as a float. Type should be `FLOAT`.