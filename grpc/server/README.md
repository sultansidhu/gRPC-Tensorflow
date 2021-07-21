# Server
Currently, all the server side code is doing, is that it is continuously polling for 
any requests from clients. Once it receives it, it encodes the model weights and layers, 
and sends them over to the client using protobuf schemes. 

The client then decodes the model, and loads up the weight data. 

## Verification
This can be checked manually. Run the server using command:
`python server.py`
And then run the client on a separate, connected machine:
`python client.py`

This should print out model accuracy on both ends. 
The server side prompt should look like:
```
313/313 - 0s - loss: 0.0789 - accuracy: 0.9742
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
Server started...
Saving model in weights.h5...
weights.h5
Time taken for returning response - 0.010860204696655273
```