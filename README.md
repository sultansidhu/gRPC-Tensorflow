# gRPC-TensorFlow
This is an experimental repository investigating the possibility of transferring tensorflow models across a network. 
## Roadmap:
- Write a model generator file (DONE)
- Write a model encoder file (DONE)
- Write a model decoder file (DONE)
- Generate protobuf service for serving this model transfer (DONE)
- Write a server file (DONE)
- Write a client file (DONE)
- Create proto encoding for model hyperparams (DONE)
- Create proto encoding for model weights (DONE)
- Write a weight encoder method within model encoder class (DONE)
- Write a weight decoder method within model decoder class (DONE)
- Encode the hyperparameters and send them over to client 
- Spawn training utilities on the client side
- Get a trained model, and test that the weights get transferred correctly (DONE)