# -*- coding:utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf8')

import tensorflow as tf
from collections import deque  # Deque used for replay memory
from tetris_fun import GameState
import numpy
import random


linesCleared = 0
MAX_REWARD = 0
reward = 0
newBoard = []

# Parameters for DQN
ACTIONS = 6  # Do nothing[0], move left[1], rotate left[2], move left[3], move right [4].
INIT_EPSILON = 1  # Starting epsilon (for exploring). This will make the agent start by choosing a random action constantly.
FINAL_EPSILON = 0.01  # Final epsilon (final % chance to take an exploring action)
OBSERVE = 20000  # Observe game for x frames. This fills the replay memory before the agent can begin training.
REPLAY_MEMORY = OBSERVE  # Size of ReplayMemory is equal to Observe because we need to populate the replaymemory before we can train from it.
BATCH_SIZE = 64  # Size of minibatch to use in training
GAMMA = 0.99  # Decay rate of past observations. Used in Q-Learning equation and dictates whether to aim for future rewards or short-sighted rewards.
LOGGING = True  # Set to True to enable logging of when a line is cleared. timeToScore.txt will be needed in .py directory.



# Creates weights for the network
def createWeight(shape):
    #  Shape is a 1-d integer array. This defines the shape of the output tensor
    print("Creating Weight")
    weight = tf.truncated_normal(shape, stddev=0.01)  # Creates a random initial weight from a standard distribution with a standard deviation of 0.01.
    return tf.Variable(weight)


# Creates biases for the network
def createBias(shape):
    print("Creating bias")
    bias = tf.constant(0.01, shape=shape)
    return tf.Variable(bias)


# Creates a convolution for the network
def createConvolution(input, filter, stride):
    # Computes a convolution given 4D input tensors (input, filter)
    return tf.nn.conv2d(input, filter, strides=[1, stride, stride, 1], padding="SAME")


# Creates max pooling layers
def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# Creates network!
def createNetwork():

    # Input is 25 x 50 x 4 (25 width, 50 height, 4 images stacked together)

    print("Creating Network...")

    # Create weight and biases for each layer
    layerOneWeights = createWeight([8, 8, 4, 32])  # Output Tensor is 8x8x4x32 for layer one. This will be the size of the layer 1 convolution.
    layerOneBias = createBias([32])

    layerTwoWeights = createWeight([4, 4, 32, 64])  # Output tensor for layer two is 4x4x32x64. This will be the size of the layer 2 convolution.
    layerTwoBias = createBias([64])

    layerThreeWeights = createWeight([3, 3, 64, 64])  # Output Tensor is 3x3x64x64 for layer three. This will be the size of the layer 3 convolution.
    layerThreeBias = createBias([64])

    weights_fc1 = createWeight([128, 128])  # Output tensor will be 512 x 512. Creates weights for fully connected ReLU layer.
    bias_fc1 = createBias([128])

    weights_fc2 = createWeight([128, ACTIONS])  # Creates weights for fullyConnectedLayer to Readout Layer.
    bias_fc2 = createBias([ACTIONS])

    # Layers created below

    # Create Input Layer
    # Input image is 25x50 and we feed in 4 images at once...
    input = tf.placeholder("float", [None, 10, 20, 4])  # Creates a tensor that will always be fed a tensor of floats 25x50x4 (input size)

    # The hidden layers will have a rectified linear activation function (ReLU)
    # Create first convolution (hidden layer one) by using the input and layerOneWeights and then adding the Bias
    conv1 = tf.nn.relu(createConvolution(input, layerOneWeights, stride=4) + layerOneBias)
    pool1 = max_pool(conv1)  # Perform max pooling

    # Create second convolution (hidden layer two) by using the first hidden layer (conv1) and the second layer weights. Add the second layer bias
    conv2 = tf.nn.relu(createConvolution(pool1, layerTwoWeights, stride=2) + layerTwoBias)

    # Create third and final convolution (hidden layer three) by using the second hidden layer (conv2) and the third layer weights and bias
    conv3 = tf.nn.relu(createConvolution(conv2, layerThreeWeights, stride=1) + layerThreeBias)

    # Reshape third layer convolution into a 1-d Tensor (basically a list or array)
    conv3Flat = tf.reshape(conv3, [-1, 128])  # Use 512 for use with weights_fc1

    hiddenFullyConnceted = tf.nn.relu(tf.matmul(conv3Flat, weights_fc1) + bias_fc1)  # Creates final hidden layer with 512 fully connected ReLU nodes

    # Create readout layer
    readout = tf.matmul(hiddenFullyConnceted, weights_fc2) + bias_fc2  # Creates readout layer

    return input, readout, hiddenFullyConnceted

# Code for training DQN and having it interact with Tetris
def trainNetwork(inputLayer, readout, fullyConnected, sess, gameState):
    global score, linesCleared, MAX_REWARD, gamesPlayed, FONT
    epsilon = INIT_EPSILON
    cycleCounter = 0

    # Dictionary to hold action selected if we want to print out what action the agent chooses

    # inputLayer is the inputLayer (duh), hiddenFullyConnected is the fully connected ReLU layer (second to last layer),
    # readout is the readout from the final layer (gives us Q Values) and sess is the TensorFlow session

    print("Training Network")

    # Define cost function
    a = tf.placeholder("float", [None, ACTIONS])  # creates a float variable that will take n x ACTIONS tensors. (Used for holding actions from minibatch)
    y = tf.placeholder("float", [None])  # Creates a float tensor that will take any shape tensor as input. (used for holding yBatch from minibatch)

    readout_action = tf.reduce_sum(tf.mul(readout, a), 1)  # multiples readout (Q values) by a (action Batch) and then computes the sum of the first row

    cost1 = tf.square(y - readout_action)  # Find the squared error...
    cost = tf.reduce_mean(cost1)  # Reduce

    trainingStep = tf.train.AdamOptimizer(0.00001).minimize(cost)  # Creates an object for training using the ADAM algorithm with a learning rate of 0.001

    replayMemory = deque()  # Will be used to store experiences. This is a First-in, First-out object

    # Save / load network
    saver = tf.train.Saver()  # Create new saver object for saving and restoring variables
    sess.run(tf.initialize_all_variables())  # Initialize all global variables
    savePoint = tf.train.get_checkpoint_state('dqn_model')  # If the checkpoint file in savedNetworks directory contains a valid CheckPointState, return it.

    # If CheckPointState exists and path exists, restore it along with the replayMemory and gameParameters
    if savePoint and savePoint.model_checkpoint_path:
        white = (255, 255, 255)
        black = (0, 0, 0)

        # Restore network weights
        saver.restore(sess=sess, save_path=savePoint.model_checkpoint_path)

        # Restore gameStats (score, games played, max reward, lines cleared)

        # Restore replayMemory and set network hyperparameters as if we were done observing/exploring
        replayMemory = numpy.load('replayMemory.npy')
        replayMemory = deque(replayMemory)  # Convert back into deque Object
        epsilon = FINAL_EPSILON
        cycleCounter = OBSERVE + 1
        print("Successfully restored previous session: " + savePoint.model_checkpoint_path)
    else:
        print("Could not load from save")

    doNothing = [1, 0, 0, 0, 0, 0]  # Send do_nothing action by default for first frame

    # imageData = image data from game, reward = received reward
    state, reward, terminal, cleared = gameState.frame_step(doNothing)

    # Stack 4 copies of the first frame into a 3D array on a new axis
    frameStack = numpy.stack((state, state, state, state), axis=2)

    localScore2 = 0
    # Run forever - this is the main code
    tot_cleared = 0
    avg_reward = 0
    while True:
        readoutEvaluated = readout.eval(feed_dict={inputLayer: [frameStack]})[0]  # readoutEvaluated is equal to the evaluation of the output layer (Q values) when feeding the input layer the newest frame

        action = numpy.zeros([ACTIONS])  # Create 1xACTIONS zeros array (for choosing action to send)
        chosenAction = 0  # Do nothing by default

        # Explore / Exploit decision
        if random.random() <= epsilon:  # If we should explore...
            #print("Exploring!")
            # Choose action randomly
            chosenAction = random.randint(0, len(action))  # Choose random action from list of actions..
            if chosenAction == len(action):
                chosenAction = chosenAction - 1  # Prevents index out of bounds as len(action) is non-zero indexed while lists are zero-indexed
            action[chosenAction] = 1  # Set that random action to 1 (for true)
        else:
            #print("Exploiting!")
            # Choose action greedily
            chosenAction = numpy.argmax(readoutEvaluated)  # Set chosenAction to the index of the largest Q-value
            action[chosenAction] = 1  # Set the largest "action" to true
            
        #print(printOutActions.get(chosenAction))  # prints the action the agent chooses at each step


        # Scale Epsilon if done observing
        if epsilon > FINAL_EPSILON and cycleCounter > OBSERVE:  # If epsilon is not final and we're done observing...
            epsilon -= 0.00002  # Subtract 0.000002 from epsilon to increase exploitation chance.

        # Run once per frame. This will send the selected action to the game and give us our reward and then train the agent with our minibatch.
        for i in range(0, 1):
            # Run selected action and observe the reward
            state, reward, terminal, cleared = gameState.frame_step(action)  # Send selected action to game and get back the game screen and the reward receieved
            tot_cleared += cleared
            avg_reward += reward
            # Print out reward received
            #print("Reward: " + str(localScore) + '  Epsilon: ' + str(epsilon))

            frame = numpy.reshape(state, (10, 20, 1))


            # Append first 3 25x50 pictures stored in framestack on new index (oldest frame = 0th index)
            frameStackNew = numpy.append(frame, frameStack[:, :, 0:3], axis=2)

            # frameStack = previous stack of frames, action = taken action, localScore = change in score (reward), frameStackNew = updated stack of frames, localTerminal = if game over
            # Only adds experience if last piece is at bottom and new piece has not been generated yet so we don't confuse agent
            replayMemory.append((frameStack, action, reward, frameStackNew))  # Store transition in replay memory as a tuple

            localScore2 = reward
            
            # If replay memory is full, get rid of oldest experience
            if len(replayMemory) > REPLAY_MEMORY:
                replayMemory.popleft()

        # If we're ready to train...
        if cycleCounter > OBSERVE and len(replayMemory) >= BATCH_SIZE:
            # Sample from replayMemory randomly
            minibatch = random.sample(replayMemory, BATCH_SIZE)
            
            # Get batch variables
            initialFrameBatch = [r[0] for r in minibatch]
            actionBatch = [r[1] for r in minibatch]
            scoreBatch = [r[2] for r in minibatch]
            updatedFrameBatch = [r[3] for r in minibatch]
            yBatch = []  # Create blank list

            batchReadout = readout.eval(feed_dict={inputLayer: updatedFrameBatch})  # Get readout of final layer (Q Values) by feeding input layer the updated frames

            for i in range(0, len(minibatch)):
                # Doubled Q-Learning, 50/50 chance of taking max or min. This reduces maximization bias.
                if random.random() < 0.49:
                    yBatch.append(scoreBatch[i] + GAMMA * numpy.max(batchReadout[i]))
                else:
                    yBatch.append(scoreBatch[i] + GAMMA * numpy.min(batchReadout[i]))

            # Perform training step by feeding ADAM optimizer the actions for the scores with the respective game state
            trainingStep.run(feed_dict={y: yBatch,
                                        a: actionBatch,
                                        inputLayer: initialFrameBatch})
        frameStack = frameStackNew  # Update Framestack
        cycleCounter += 1
        if cycleCounter % 500 == 0:
            rollout_avg_reward = float(avg_reward) / float(500)
            avg_reward = 0
            print("global step:{}".format(cycleCounter))
            print("500 steps avarage reward:{}".format(str(rollout_avg_reward)))

        # Uncomment to print the frame we're on and the Q-Values for the current game state
        #print('Frame: ' + str(cycleCounter) + '  Q-Values: ' + str(readoutEvaluated))

        # Save network every 50000 steps
        if cycleCounter % 50000 == 0:
            saver.save(sess, 'dqn_model/Tetris-dqn', global_step=cycleCounter)  # Save network weights to directory savedNetworks
            numpy.save('replayMemory', replayMemory)  # Save replay memory

        if terminal:
            gameState.reinit()
            frameStack = gameState.get_state()
            tot_cleared = 0
            avg_reward = 0

if __name__ == "__main__":
    config = tf.ConfigProto()  
    config.gpu_options.allow_growth=True  
    sess = tf.InteractiveSession(config=config)

    gameState = GameState()
    input, readout, fullyConnected = createNetwork()

    initVars = tf.initialize_all_variables()
    trainNetwork(input, readout, fullyConnected, sess, gameState)
