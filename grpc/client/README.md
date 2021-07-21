# Client 
Currently, all the client side code is a single request sent to the gRPC server
that asks the server to encode the model and weights, and send it over.

The client then decodes the model, and loads up the weight data. 

## Verification
This can be checked manually. Make sure that you have a server running, and run the client file:
`python client.py`
This should print out the exact same accuracy as the server side machine.

The sample prompt for client should look like:
```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
flatten (Flatten)            (None, 784)               0         
_________________________________________________________________
dense (Dense)                (None, 128)               100480    
_________________________________________________________________
dropout (Dropout)            (None, 128)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 10)                1290      
=================================================================
Total params: 101,770
Trainable params: 101,770
Non-trainable params: 0
_________________________________________________________________
None
2021-07-21 21:57:47.553833: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:176] None of the MLIR Optimization Passes are enabled (registered 2)
2021-07-21 21:57:47.554292: I tensorflow/core/platform/profile_utils/cpu_utils.cc:114] CPU Frequency: 2249995000 Hz
313/313 - 0s - loss: 0.0789 - accuracy: 0.9742
```