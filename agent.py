import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque


import torch

def huber_loss(y_true, y_pred, clip_delta=1.0):
    """Huber loss - Custom Loss Function for Q Learning
    """
    error = y_true - y_pred
    abs_error = error.abs()
    mask = abs_error <= clip_delta
    squared_loss = 0.5 * error.pow(2)
    clip_delta_tensor = torch.tensor(clip_delta)
    quadratic_loss = 0.5 * clip_delta_tensor.pow(2) + clip_delta * (abs_error - clip_delta)
    return torch.where(mask, squared_loss, quadratic_loss).mean()


import torch.nn as nn

class Model(nn.Module):
    """The neural network model"""
    
    def __init__(self, state_size, action_size, hidden_size=32, num_layers=10):
        super(Model, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=state_size, hidden_size=hidden_size, num_layers=num_layers)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, action_size)
    
    def forward(self, x, h0=None):
        x, (hn, cn) = self.lstm(x, h0)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.fc5(x)


class Agent:
    """ Stock Trading Bot """
    
    def __init__(self, state_size, model_name=None):

        # agent attributes
        self.state_size = state_size    	# normalized previous days
        self.action_size = 3           		# [sit, buy, sell]
        self.model_name = model_name
        self.inventory = []
        self.memory = deque(maxlen=100000)
        self.first_iter = True
        # generate model
        self.model_name = model_name
        
        try:
            # load model from memory if it exists
            self.model = torch.load(str(self.model_name + ".pt"))
        except:
            # create model
            self.model = Model(self.state_size, self.action_size)
            
        self.gamma = 0.95 # affinity for long term reward
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.loss = huber_loss
        self.custom_objects = {"huber_loss": huber_loss}  # important for loading the model from memory
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            
            
    def get_status(self):
        """Get the status of the agent (inventory, epsilon, etc.)
        Explanation of the status:
        - gamma: discount factor for future rewards 
        - epsilon: exploration rate
        - epsilon_min: minimum exploration rate
        - epsilon_decay: decay rate of exploration rate
        - learning_rate: learning rate
        - loss: loss function
        - optimizer: optimizer
        - model: model
        - len_memory: length of memory
        - len_inventory: length of inventory
        """
        return {
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "learning_rate": self.learning_rate,
            "loss": self.loss,
            "optimizer": self.optimizer,
            "model": self.model,
            "len_memory": len(self.memory),
            "len_inventory": len(self.inventory),

        }
        
    
    def remember(self, state, action, reward, next_state, done):
        """Adds relevant data to memory
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def get_action(self, state, is_eval=False):
        """Take action from given possible set of actions
        """
        # Take a random action in order to diversify experience at the beginning
        if not is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        if self.first_iter:
            self.first_iter = False
            return 1 # make a definite buy on the first iter

        # Compute the action probabilities
        action_probs = self.model(state)
        # Return the action with the highest probability
        return action_probs.argmax().item()


    def replay(self, batch_size):
        """Train on previous experiences in memory
        """
        # Sample a mini-batch from the memory
        mini_batch = random.sample(self.memory, batch_size)
        # Initialize the training data for the mini-batch
        X_train, y_train = [], []
        # Iterate over the mini-batch
        for state, action, reward, next_state, done in mini_batch:
            # Compute the target for the current (state, action) pair
            if done:
                target = reward
            else:
                # approximate deep q-learning equation
                target = reward + self.gamma * torch.max(self.model(next_state)[0])
            
            # Estimate the q-values for the current state
            q_values = self.model(state)

            # Update the target for the current action based on discounted reward
            q_values[0][action] = target
            X_train.append(state[0])
            y_train.append(q_values[0])

        # Convert the training data to tensors
        #create a tensor from a list of tensors
        X_train = torch.cat(X_train, dim=0)
        #reshape the tensor
        X_train = X_train.view(-1, self.state_size)
        y_train = torch.cat(y_train, dim=0)
        y_train = y_train.view(-1, self.action_size)
        # Clear the gradients
        self.optimizer.zero_grad()
        # Compute the loss
        loss = self.loss(self.model(X_train), y_train)
        # Backpropagate the loss
        loss.backward()
        # Update the model's parameters
        self.optimizer.step()
        # Return the loss
        return loss.item()

    def save_model(self):
        """Saves the model"""
        torch.save(self.model, str(self.model_name + ".pt"))