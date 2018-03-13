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
    - param[0]: `W_{xi} W_{xf} W_{xo} W_{xg}`
    - param[1]: `b_i b_f b_o b_g`
    - param[2]: `W_{hi} W_{hf} W_{ho} W_{hg}`
* tensorflow format of weights
    - variables[0]: `[W_{xi} W_{hi}] [W_{xg} W_{hg}] [W_{xf} W_{hf}] [W_{xo} W_{ho}]`
    - variables[1]: `b_i b_g b_f b_o`