# coding: utf-8
"""Define the Queue environment from problem 3 here."""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import sys
from gym import Env, spaces
from gym.envs.registration import register


class QueueEnv(Env):
    """Implement the Queue environment from problem 3.

    Parameters
    ----------
    p1: float
      Value between [0, 1]. The probability of queue 1 receiving a new item.
    p2: float
      Value between [0, 1]. The probability of queue 2 receiving a new item.
    p3: float
      Value between [0, 1]. The probability of queue 3 receiving a new item.

    Attributes
    ----------
    nS: number of states
    nA: number of actions
    P: environment model
    """
    metadata = {'render.modes': ['human']}

    SWITCH_TO_1 = 0
    SWITCH_TO_2 = 1
    SWITCH_TO_3 = 2
    SERVICE_QUEUE = 3

    def __init__(self, p1, p2, p3):
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.MultiDiscrete([(1, 3), (0, 5), (0, 5), (0, 5)])
        self.receivingProb = {0:p1,1:p2,2:p3}
        self.nS = 3*6*6*6 
        self.nA = 4 
        self.P = {}# {s : {a : [] for a in range(nA)} for s in range(nS)}
        self.s = (1,0,0,0) #???
        # case: read current and it's current
        # 0-4 items get a new item with probability -> transition to state 
        # 1-5 items get reward & delete an item
        # ---> prob p1 to remain same state for queues with 1-5 items .
        #            1-p1 to transit to prev state for queues with 1-5 items.
        #           p1 to transit to next state for queues with 0 item.  
        #           1-p1 to remain same state for queues with 0 item. 
        # case: others 
        # ---> prob p1 to transit to next state for queues with 0-4 items. 
        #           1-p1 to remain the same state for queues with 0-4 items.
        #           1 to remain the same state for queues with 5 items.  

        # read on current queue
        for (q1,q2,q3) in [(x,y,z) for x in range(6) for y in range(6) for z in range(6)]:
            for act in range(1,4):
                self.P[(act,q1,q2,q3)] = {3:[]}
                if (act==1 and q1 == 0) or (act==2 and q2==0) or (act==3 and q3==0):
                    self.P[(act,q1,q2,q3)][3] = \
                        [(p1*p2*p3, (act,self.add(q1,1),self.add(q2,1),self.add(q3,1)), 0.0, False), \
                         ((1-p1)*p2*p3, (act,q1,self.add(q2,1),self.add(q3,1)), 0.0, False), \
                         (p1*(1-p2)*p3, (act,self.add(q1,1),q2,self.add(q3,1)), 0.0, False), \
                         (p1*p2*(1-p3), (act,self.add(q1,1),self.add(q2,1),q3), 0.0, False), \
                         ((1-p1)*(1-p2)*p3, (act,q1,q2,self.add(q3,1)), 0.0, False), \
                         ((1-p1)*p2*(1-p3), (act,q1,self.add(q2,1),q3), 0.0, False), \
                         (p1*(1-p2)*(1-p3), (act,self.add(q1,1),q2,q3), 0.0, False), \
                         ((1-p1)*(1-p2)*(1-p3), (act,q1,q2,q3), 0.0, False)]
                elif act == 1:
                     self.P[(act,q1,q2,q3)][3] = \
                        [(p1*p2*p3, (act,self.add(q1,1-1),self.add(q2,1),self.add(q3,1)), 1.0, False), \
                         ((1-p1)*p2*p3, (act,q1-1,self.add(q2,1),self.add(q3,1)), 1.0, False), \
                         (p1*(1-p2)*p3, (act,self.add(q1,1-1),q2,self.add(q3,1)), 1.0, False), \
                         (p1*p2*(1-p3), (act,self.add(q1,1-1),self.add(q2,1),q3), 1.0, False), \
                         ((1-p1)*(1-p2)*p3, (act,q1-1,q2,self.add(q3,1)), 1.0, False), \
                         ((1-p1)*p2*(1-p3), (act,q1-1,self.add(q2,1),q3), 1.0, False), \
                         (p1*(1-p2)*(1-p3), (act,self.add(q1,1-1),q2,q3), 1.0, False), \
                         ((1-p1)*(1-p2)*(1-p3), (act,q1-1,q2,q3), 1.0, False)]
                elif act == 2:
                     self.P[(act,q1,q2,q3)][3] = \
                        [(p1*p2*p3, (act,self.add(q1,1),self.add(q2,1-1),self.add(q3,1)), 1.0, False), \
                         ((1-p1)*p2*p3, (act,q1,self.add(q2,1-1),self.add(q3,1)), 1.0, False), \
                         (p1*(1-p2)*p3, (act,self.add(q1,1),q2-1,self.add(q3,1)), 1.0, False), \
                         (p1*p2*(1-p3), (act,self.add(q1,1),self.add(q2,1-1),q3), 1.0, False), \
                         ((1-p1)*(1-p2)*p3, (act,q1,q2-1,self.add(q3,1)), 1.0, False), \
                         ((1-p1)*p2*(1-p3), (act,q1,self.add(q2,1-1),q3), 1.0, False), \
                         (p1*(1-p2)*(1-p3), (act,self.add(q1,1),q2-1,q3), 1.0, False), \
                         ((1-p1)*(1-p2)*(1-p3), (act,q1,q2-1,q3), 1.0, False)]
                elif act == 3:
                     self.P[(act,q1,q2,q3)][3] = \
                        [(p1*p2*p3, (act,self.add(q1,1),self.add(q2,1),self.add(q3,1-1)), 1.0, False), \
                         ((1-p1)*p2*p3, (act,q1,self.add(q2,1),self.add(q3,1-1)), 1.0, False), \
                         (p1*(1-p2)*p3, (act,self.add(q1,1),q2,self.add(q3,1-1)), 1.0, False), \
                         (p1*p2*(1-p3), (act,self.add(q1,1),self.add(q2,1),q3-1), 1.0, False), \
                         ((1-p1)*(1-p2)*p3, (act,q1,q2,self.add(q3,1-1)), 1.0, False), \
                         ((1-p1)*p2*(1-p3), (act,q1,self.add(q2,1),q3-1), 1.0, False), \
                         (p1*(1-p2)*(1-p3), (act,self.add(q1,1),q2,q3-1), 1.0, False), \
                         ((1-p1)*(1-p2)*(1-p3), (act,q1,q2,q3-1), 1.0, False)]


        for (q_from, q_to) in [(x,y) for x in range(1,4) for y in range(1,4)]:
            for (q1,q2,q3) in [(x,y,z) for x in range(6) for y in range(6) for z in range(6)]:
                if q_to-1 not in self.P[(q_from,q1,q2,q3)]:
                     self.P[(q_from,q1,q2,q3)][q_to-1] = \
                        [(p1*p2*p3, (q_to,self.add(q1,1),self.add(q2,1),self.add(q3,1)), 0.0, False), \
                         ((1-p1)*p2*p3, (q_to,q1,self.add(q2,1),self.add(q3,1)), 0.0, False), \
                         (p1*(1-p2)*p3, (q_to,self.add(q1,1),q2,self.add(q3,1)), 0.0, False), \
                         (p1*p2*(1-p3), (q_to,self.add(q1,1),self.add(q2,1),q3), 0.0, False), \
                         ((1-p1)*(1-p2)*p3, (q_to,q1,q2,self.add(q3,1)), 0.0, False), \
                         ((1-p1)*p2*(1-p3), (q_to,q1,self.add(q2,1),q3), 0.0, False), \
                         (p1*(1-p2)*(1-p3), (q_to,self.add(q1,1),q2,q3), 0.0, False), \
                         ((1-p1)*(1-p2)*(1-p3), (q_to,q1,q2,q3), 0.0, False)]
 
        for s in self.P:
            for act in self.P[s]:
                new_states = {}
                for tr in self.P[s][act]:
			        new_states[(tr[1],tr[2])] = new_states.get((tr[1],tr[2]),0)+tr[0]
                new_transit = []
                for tr in new_states:
                    new_transit.append((new_states[tr],tr[0],tr[1],False))
                self.P[s][act] = new_transit

    def add(self, q, n):
        if n >= 0:
            return min(q+n,5)
        else:
            return max(q+n,0)

    def _reset(self):
        """Reset the environment.

        The server should always start on Queue 1.

        Returns
        -------
        (int, int, int, int)
          A tuple representing the current state with meanings
          (current queue, num items in 1, num items in 2, num items in
          3).
        """
        self.s[0] = 1
        return self.s # should be elements cleaned up? 

    def _step(self, action):
        """Execute the specified action.

        Parameters
        ----------
        action: int
          A number in range [0, 3]. Represents the action.

        Returns
        -------
        (state, reward, is_terminal, debug_info)
          State is the tuple in the same format as the reset
          method. Reward is a floating point number. is_terminal is a
          boolean representing if the new state is a terminal
          state. debug_info is a dictionary. You can fill debug_info
          with any additional information you deem useful.
        """
        # how to get the current state? 
        prob = random.random()
        for (p, nextState, reward, is_terminal) in self.P[self.s][action]:
            if prob <= p:
                self.s = nextState # ???
                return (nextState, reward, is_terminal, {}) 
            prob -= p   
        return None, None, None, None

    def _render(self, mode='human', close=False):
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        queues = ["" for x in range(3)]
        for i in range(3):
            queues[i] += "> " if self.s[0]==i+1 else "  "
            queues[i] += "o"*self.s[i+1]
            outfile.write(queues[i])
        return outfile
        #pass

    def _seed(self, seed=None):
        """Set the random seed.

        Parameters
        ----------
        seed: int, None
          Random seed used by numpy.random and random.
        """
        pass

    def query_model(self, state, action):
        """Return the possible transition outcomes for a state-action pair.

        This should be in the same format at the provided environments
        in section 2.

        Parameters
        ----------
        state
          State used in query. Should be in the same format at
          the states returned by reset and step.
        action: int
          The action used in query.

        Returns
        -------
        [(prob, nextstate, reward, is_terminal), ...]
          List of possible outcomes
        """
        return self.P[state][action]

        #return None

    def get_action_name(self, action):
        if action == QueueEnv.SERVICE_QUEUE:
            return 'SERVICE_QUEUE'
        elif action == QueueEnv.SWITCH_TO_1:
            return 'SWITCH_TO_1'
        elif action == QueueEnv.SWITCH_TO_2:
            return 'SWITCH_TO_2'
        elif action == QueueEnv.SWITCH_TO_3:
            return 'SWITCH_TO_3'
        return 'UNKNOWN'


register(
    id='Queue-1-v0',
    entry_point='deeprl_hw1.queue_envs:QueueEnv',
    kwargs={'p1': .1,
            'p2': .9,
            'p3': .1})

register(
    id='Queue-2-v0',
    entry_point='deeprl_hw1.queue_envs:QueueEnv',
    kwargs={'p1': .1,
            'p2': .1,
            'p3': .1})
