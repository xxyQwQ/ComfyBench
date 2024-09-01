- `CM_Vec2BinaryOperation`: The node performs binary operations on two-dimensional vectors, producing a new vector as a result. It abstracts mathematical operations that combine two vectors into one, based on a specified operation.
    - Parameters:
        - `op`: Specifies the binary operation to be performed on the vectors. It determines how the two input vectors will be combined. Type should be `COMBO[STRING]`.
    - Inputs:
        - `a`: The first two-dimensional vector involved in the binary operation. Type should be `VEC2`.
        - `b`: The second two-dimensional vector involved in the binary operation. Type should be `VEC2`.
    - Outputs:
        - `vec2`: The resulting two-dimensional vector after applying the specified binary operation on the input vectors. Type should be `VEC2`.