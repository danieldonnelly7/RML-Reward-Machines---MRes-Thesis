import numpy as np
from enum import Enum
import random

class Actions(Enum):
    up    = 0 # move up
    right = 1 # move right
    down  = 2 # move down
    left  = 3 # move left
    none  = 4 # none or pick
    drop  = 5


class LetterEnv:

    def __init__(self, n = 3):
        self.map_height, self.map_width = 6, 6
        self._load_map()
        self.additional_states()
        self.max_n = n
        self.n = random.randint(1,self.max_n)
        self.monitor_state = [0]*20
        self.evaluation_n = False   # Used for evaluation to set n = to max_n value

    def additional_states(self):
        self.state_objects = {}
        for key, value in self.objects.items():
            self.state_objects[key] = value
        self.object_list = list(self.state_objects.values())
        self.object_list.append('b')  # Order will be [a,c,d,b]

    def _load_map(self):
        # Creating the map
        self.objects = {}
        self.objects[(1,4)] = "a"   # Changes to b after n steps
        self.objects[(4,4)] = "c"
        self.objects[(4,1)] = "d"

        self.forbidden_transitions = set()
        for x in range(self.map_width):
            self.forbidden_transitions.add((x,0,Actions.down)) 
            self.forbidden_transitions.add((x,5,Actions.up)) 
        for y in range(self.map_height):
            self.forbidden_transitions.add((0,y,Actions.left))
            self.forbidden_transitions.add((5,y,Actions.right))

        self.actions = [Actions.up.value,Actions.right.value,Actions.down.value,Actions.left.value]

    def get_true_propositions(self):
            """
            Returns the string with the propositions that are True in this state
            """
            ret = ""
            if self.agent in self.objects:
                ret += self.objects[self.agent]
            if ret == "a":
                self.n_steps +=1
            return ret

    def execute_action(self, a):
            """
            We execute 'action' in the game
            """
            x,y = self.agent
            self.agent = self._get_new_position(x,y,a)

    def _get_new_position(self, x, y, a):
            action = Actions(a)
            # executing action
            if (x,y,action) not in self.forbidden_transitions:
                if action == Actions.up   : y+=1
                if action == Actions.down : y-=1
                if action == Actions.left : x-=1
                if action == Actions.right: x+=1
            return x,y

    def get_features(self):
            """
            Returns the features of the current state (i.e., the location of the agent)
            """
            x,y = self.agent
            if self.n_steps >= self.n:
                self.objects[(1,4)] = "b"   # Changes a to b after self.n steps have passed

            self.one_hot_objects = self.get_additional_state_vector()
            return np.array([x,y] + self.one_hot_objects, dtype=int)

    def get_additional_state_vector(self):
        self.true_props = self.get_true_propositions()
        new_objects = [0]*len(self.object_list)    # The new list for the one hot encoding

        if len(self.true_props) > 0:   # Adding 1s to where currently possesed objects are
            if self.true_props in self.object_list:
                encoding_index = self.object_list.index(self.true_props)
                new_objects[encoding_index] = 1
        if isinstance(self.monitor_state, np.ndarray):     # Logic changes depending on whether the monitor state is a list or numpy array
            self.monitor_state = self.monitor_state.tolist()
            new_objects = new_objects + self.monitor_state
        elif isinstance(self.monitor_state, list):
            new_objects = new_objects + self.monitor_state
        else:
            new_objects.append(self.monitor_state)    # Adding in the monitor state to the states list

        return new_objects
    
    def reset(self, seed=None, options=None):
        if self.evaluation_n == True:
             self.n = self.max_n
        else:
            self.n = random.randint(1,self.max_n)
        self.agent = (0,0)
        self.objects[(1,4)] = "a"   # resets to a
        self.true_props = self.get_true_propositions()
        self.one_hot_objects = [0]*len(self.object_list)
        #self.monitor_state = [0]*20
        self.one_hot_objects = self.get_additional_state_vector()
        self.n_steps = 0

    def get_model(self):
        """
        This method returns a model of the environment. 
        We use the model to compute optimal policies using value iteration.
        The optimal policies are used to set the average reward per of each task to 1.
        """
        S = [(x,y) for x in range(self.map_width) for y in range(self.map_height)] # States
        A = self.actions.copy() # Actions
        L = self.objects.copy() # Labeling function
        T = {}                  # Transitions (s,a) -> s' (they are deterministic)
        for s in S:
            x,y = s
            for a in A:
                T[(s,a)] = self._get_new_position(x,y,a)
        return S,A,L,T # SALT xD
    
    def show(self):
        # Print the top boundary
        print("+" + "-" * self.map_width + "+")
        
        for y in range(self.map_height):
            # Print the left boundary
            print("|", end="")
            for x in range(self.map_width):
                if (x, y) == self.agent:
                    print("x", end="")  # Print agent
                elif (x, y) in self.objects:
                    print(self.objects[(x, y)], end="")  # Print objects
                else:
                    print(" ", end="")  # Print empty space
            # Print the right boundary
            print("|")
        
        # Print the bottom boundary
        print("+" + "-" * self.map_width + "+")


    def get_monitor_state(self, state):
        self.monitor_state = state

            
class LetterEnv_RNN(LetterEnv):
    def get_additional_state_vector(self):
        self.true_props = self.get_true_propositions()
        new_objects = [0]*len(self.object_list)    # The new list for the one hot encoding

        if len(self.true_props) > 0:   # Adding 1s to where currently possesed objects are
            if self.true_props in self.object_list:
                encoding_index = self.object_list.index(self.true_props)
                new_objects[encoding_index] = 1

        return new_objects



