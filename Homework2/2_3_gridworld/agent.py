import util, random
import numpy as np


class Agent:

    def getAction(self, state):
        """
        For the given state, get the agent's chosen
        action.  The agent knows the legal actions
        """
        abstract

    def getValue(self, state):
        """
        Get the value of the state.
        """
        abstract

    def getQValue(self, state, action):
        """
        Get the q-value of the state action pair.
        """
        abstract

    def getPolicy(self, state):
        """
        Get the policy recommendation for the state.

        May or may not be the same as "getAction".
        """
        abstract

    def update(self, state, action, nextState, reward):
        """
        Update the internal state of a learning agent
        according to the (state, action, nextState)
        transistion and the given reward.
        """
        abstract


class RandomAgent(Agent):
    """
    Clueless random agent, used only for testing.
    """

    def __init__(self, actionFunction):
        self.actionFunction = actionFunction

    def getAction(self, state):
        return random.choice(self.actionFunction(state))

    def getValue(self, state):
        return 0.0

    def getQValue(self, state, action):
        return 0.0

    def getPolicy(self, state):
        return "random"

    def update(self, state, action, nextState, reward):
        pass


################################################################################
# Exercise 4


class ValueIterationAgent(Agent):

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
        Your value iteration agent should take an mdp on
        construction, run the indicated number of iterations
        and then act according to the resulting policy.
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        threshold = 0.001

        self.states = mdp.getStates()  # states are tuples (i, j) on the grid

        self.value = {s: 0 for s in self.states}
        self.qvalue = {
            (s, a): 0 for s in self.states for a in mdp.getPossibleActions(s)
        }

        for iters in range(self.iterations):
            self.previous_value = self.value.copy()
            print(iters)
            print(self.value)

            for state in self.states:
                actions = mdp.getPossibleActions(state)
                if len(actions) != 0:
                    for action in actions:
                        self.qvalue[(state, action)] = 0

                        model = self.mdp.getTransitionStatesAndProbs(state, action)

                        for s_, p in model:
                            self.qvalue[state, action] += (
                                p * self.value[s_]
                            )  # Bellman update

                        self.qvalue[state, action] = (
                            self.mdp.getReward(state, action, None)
                            + self.discount * self.qvalue[state, action]
                        )

                    self.value[state] = np.max(
                        [
                            self.qvalue[state, a]
                            for a in self.mdp.getPossibleActions(state)
                        ]
                    )

            unchanged = True
            for key in self.value:
                if abs(self.value[key] - self.previous_value[key]) > threshold:
                    unchanged = False
                    break

            if unchanged:
                break
        print("Number of iterations:", iters + 1)

    def getValue(self, state):
        """
        Look up the value of the state (after the indicated
        number of value iteration passes).
        """
        return self.value[state]

    def getQValue(self, state, action):
        """
        Look up the q-value of the state action pair
        (after the indicated number of value iteration
        passes).  Note that value iteration does not
        necessarily create this quantity and you may have
        to derive it on the fly.
        """
        return self.qvalue[(state, action)]

    def getPolicy(self, state):
        """
        Look up the policy's recommendation for the state
        (after the indicated number of value iteration passes).
        """
        if len(self.mdp.getPossibleActions(state)) != 0:
            # print([self.qvalue[state, a] for a in self.mdp.getPossibleActions(state)])

            maxqval = np.inf * -1
            for a in self.mdp.getPossibleActions(state):
                if self.qvalue[state, a] > maxqval:
                    maxqval = self.qvalue[state, a]
                    maxaction = a
            return maxaction

    def getAction(self, state):
        """
        Return the action recommended by the policy.
        """
        return self.getPolicy(state)

    def update(self, state, action, nextState, reward):
        """
        Not used for value iteration agents!
        """
        pass


################################################################################
# Below can be ignored for Exercise 4


class QLearningAgent(Agent):

    def __init__(self, actionFunction, discount=0.9, learningRate=0.1, epsilon=0.2):
        """
        A Q-Learning agent gets nothing about the mdp on
        construction other than a function mapping states to actions.
        The other parameters govern its exploration
        strategy and learning rate.
        """
        self.setLearningRate(learningRate)
        self.setEpsilon(epsilon)
        self.setDiscount(discount)
        self.actionFunction = actionFunction

        raise "Your code here."

    # THESE NEXT METHODS ARE NEEDED TO WIRE YOUR AGENT UP TO THE CRAWLER GUI

    def setLearningRate(self, learningRate):
        self.learningRate = learningRate

    def setEpsilon(self, epsilon):
        self.epsilon = epsilon

    def setDiscount(self, discount):
        self.discount = discount

    # GENERAL RL AGENT METHODS

    def getValue(self, state):
        """
        Look up the current value of the state.
        """

        raise ValueError("Your code here.")

    def getQValue(self, state, action):
        """
        Look up the current q-value of the state action pair.
        """

        raise ValueError("Your code here.")

    def getPolicy(self, state):
        """
        Look up the current recommendation for the state.
        """

        raise ValueError("Your code here.")

    def getAction(self, state):
        """
        Choose an action: this will require that your agent balance
        exploration and exploitation as appropriate.
        """

        raise ValueError("Your code here.")

    def update(self, state, action, nextState, reward):
        """
        Update parameters in response to the observed transition.
        """

        raise ValueError("Your code here.")
