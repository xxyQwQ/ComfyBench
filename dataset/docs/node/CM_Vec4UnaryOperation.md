- `CM_Vec4UnaryOperation`: This node performs unary operations on 4-dimensional vectors, applying a specified operation to a single Vec4 input and producing a Vec4 output. It abstracts complex vector manipulations into simple, callable operations, facilitating mathematical computations on 4D vectors.
    - Parameters:
        - `op`: Specifies the unary operation to be performed on the Vec4 input. The choice of operation directly influences the result, enabling a variety of mathematical manipulations. Type should be `COMBO[STRING]`.
    - Inputs:
        - `a`: The Vec4 input on which the unary operation is to be performed. This vector serves as the primary data for the operation, determining the nature of the computation. Type should be `VEC4`.
    - Outputs:
        - `vec4`: The result of applying the specified unary operation on the Vec4 input, returned as a Vec4. This output encapsulates the transformed vector, reflecting the mathematical computation performed. Type should be `VEC4`.