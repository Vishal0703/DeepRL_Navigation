
The agent has been trained using 3 different learning algorithms
1. DQN
2. DDQN
3. Duelling DQN


The common hyperparameters for the 3 algorithms are:
    
    BUFFER_SIZE = int(1e4)  # replay buffer size
    BATCH_SIZE = 64         # minibatch size
    GAMMA = 0.99            # discount factor
    TAU = 1e-3              # for soft update of target parameters
    LR = 5e-4               # learning rate 
    UPDATE_EVERY = 4        # how often to update the network
    

The model used for DQN and DDQN algorithms is a 4 layered NN with 2 hidden layers 
each having 256 units with relu activations, whereas on the last layer no activations have been applied.
    

For Duelling DQN, a bifurcation at the last layer has been made to compute V(s) and A(s,a) respectively.

They have been merged using V(s) + A(s,a) - mean(A(s,a))

The NN parameters have also been changed to 128 units in each hidden layer.

The number of episodes taken by the different algorithms respectively are:

1. 496
2. 407
3. 362

The plots of reward vs episodes have been saved as 

1. nav.png
2. nav_ddqn.png
3. nav_dueldqn.png
