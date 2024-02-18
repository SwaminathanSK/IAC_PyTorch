My implementation of IAC in PyTorch following the paper - In-sample Actor Critic for Offline Reinforcement Learning (https://openreview.net/pdf?id=dfDv0WU853R)

#### Key contributions:
-  Developed an actor-critic + sampling-importance-resampling framework for continuous control tasks in MuJoCo
-  Integrated Sum-tree datastructure to store memory for resampling at each training step
-  Integrated target-network update and the rest of the above changes onto an existing AWR open-source PyTorch implementation

#### Issues and Future directions:
-  Logging of average returns to be fixed
-  Implementation to be proof-read for correctness and execution of the algorithm
-  Integrate discrete control environment support (Something that the original paper does not accomplish)
