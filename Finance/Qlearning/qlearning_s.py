def shift(arr, n):
    """ shift a one ndarray"""
    e = np.empty_like(arr)
    if n >= 0:
        e[:n] = arr[0]
        e[n:] = arr[:-n]
    else:
        e[n:] = arr[0]
        e[:n] = arr[-n:]
    return e

def sharp_ratio(r):
    """ Return sharp ratio of returns """
    return np.mean(r) / np.std(r)


class QlearningStock(object):
    def __init__(self, arr):
        self.arr = arr
        self.market_returns = shift(self.arr,1)/self.arr - 1
        self.samples = len(self.arr)
        self.actions = np.array([1,-1]) #long, short
        self.states = np.array([0,1])
        self.nb_actions = len(self.actions)
        self.nb_states = 2
        self.epsilon = 0.05
        self.risk_free_rate = 0
        self.transaction_cost = 0
        # self.alpha = 0.05
        self.utility_function = sharp_ratio
        self.gamma = 0.95 # discount factor
        self.nmc = 20 # nb of simulations of Monte Carlo
        self.states = np.zeros(self.samples)
        self.q = np.zeros((self.nb_states,self.nb_actions))
        self.rewards = np.zeros(self.samples)
        self.nb_visits = np.zeros((self.nb_states, self.nb_actions))
        self.current_state = np.random.choice(self.states)

    def reset(self):
        self.nb_visits = np.zeros((self.nb_states, self.actions))
        self.current_state = self.random.choice([0,1])

    def get_reward(self, state, next_state, market_return):
        """ Get reward associated at time t"""
        reward =  (1 + self.risk_free_rate * (1-state) + market_return * state  ) * (
        1 - self.transaction_cost * np.abs(next_state - state))
        return reward

#     def update_reward(self, reward, t):
#         """ Update reward at time t"""
#         self.rewards = reward[t]

    def simu_transition(self, action):
        """ Returns simulation of next state based on action and previous state"""
        if action == 1:
            return 1
        if action == -1:
            return 0


    def get_action(self, state):
        """ Select actions with epison greedy Q learning """
        #Epsilon-Greedy
        if np.random.random() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            return 2*np.argmax(self.q[state,:]) -1 # to modify

    @staticmethod
    def action_to_postion(action):
        """action to position mapping for the Q matrix"""
        if action == 1:
            return 1
        if action == -1:
            return 0



    def qlearning_step(self, verbose=True):
        for i in range(1, self.samples):
            market_return = self.market_returns[i]
            a = self.get_action(self.current_state)  # random action
            a_position = self.action_to_postion(a)
            print("actions : {}".format(a))
            self.nb_visits[self.current_state, a_position] += 1  # update nb_visits
            # simulate transition
            new_state = self.simu_transition(a)
            # get immediate reward
            print(self.current_state, new_state,a)
            self.reward = self.get_reward(self.current_state,new_state,market_return)
            # put reward in rewards to compute utility function
            print(self.reward)
            self.rewards[i] = self.reward
            # Calculate utility function
            u = self.utility_function(self.rewards[:i+1]) # compute sharp ration until i
            print(u)
            # Compute temporal difference
            td = u + self.gamma * (self.q[new_state, :].max()) - self.q[self.current_state, a_position]
            # Print infos
            if verbose:
                print("state : {}, new_state : {}, actions : {}, reward :{}, ".format(
                        self.current_state,new_state,a,self.reward))
                print("Q matrix : \n {}".format(self.q))
            self.q[self.current_state, a_position] +=  + (1.0 / self.nb_visits[self.current_state, a_position]) * td
            self.states[i] = new_state
            self.current_state = new_state

    def monte_carlo(self, verbose=False):
        """ Perform monte carlo iteration for Q learning """
        for i in range(self.nmc):
            print("Simulation nb : {}".format(i))
            self.qlearning_step(verbose=verbose)
            self.reset() # reset nb_visits and starting point 
