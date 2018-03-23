# Batch Normalization
 
We can merge a conv layer followed by a batch normalization layer into one single conv layer.

The equation of a conv layer followed by a batch normalization layer: 
- `y_conv = Wx + b`
- `y = gamma * (y_conv - mean) / std_dev + beta`.

Let `W' = gamma * W / std_dev` and `b' = gamma * (b - mean) / std_dev + beta`, then `y = W'x + b'`. 

# LSTM layer

* equations of LSTM
    ```
    i_t := \sigmoid[ W_{hi} * h_{t-1} + W_{xi} * x_t + b_i ]
    f_t := \sigmoid[ W_{hf} * h_{t-1} + W_{xf} * x_t + b_f ]
    o_t := \sigmoid[ W_{ho} * h_{t-1} + W_{xo} * x_t + b_o ]
    g_t :=    \tanh[ W_{hg} * h_{t-1} + W_{xg} * x_t + b_g ]
    c_t := (f_t .* c_{t-1}) + (i_t .* g_t)
    h_t := o_t .* \tanh[c_t]
    ```

* caffe format of weights
    - params[0]: `W_{xi} W_{xf} W_{xo} W_{xg}`
    - params[1]: `b_i b_f b_o b_g`
    - params[2]: `W_{hi} W_{hf} W_{ho} W_{hg}`
* tensorflow format of weights
    - variables[0]: `[W_{xi} W_{hi}] [W_{xg} W_{hg}] [W_{xf} W_{hf}] [W_{xo} W_{ho}]`
    - variables[1]: `b_i b_g b_f b_o`

# Multilayer Bidirectional LSTM

The flipped input (`flip(x)`) along the sequence axis is fed into the backward LSTM layer.  

The input of the next forward and backward LSTM layer is expressed as `x_{n+1} = concat(L_fw_n(x_n), flip(L_bw_n(flip(x_n))))`.