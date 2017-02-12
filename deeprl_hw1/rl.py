# coding: utf-8
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np
from ipdb import set_trace as debug

def evaluate_policy(env, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """Evaluate the value of a policy.

    See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray
      The value for the given policy
    """

    # initialize
    V = np.zeros(env.nS)

    # value iteration follow given policy
    for i in range(max_iterations):
      
      preV = V.copy()

      # calculate new value function: one-step look ahead 
      V = np.zeros(env.nS)
      for s in range(env.nS):  
        for prob, nextstate, reward, is_terminal in env.P[s][policy[s]]:
          V[s] += prob*(reward+gamma*preV[nextstate])

      # terminate condition check
      diff = np.max(np.abs(V-preV))
      if diff < tol:
        break

    return V

def value_function_to_policy(env, gamma, value_func):
    """Output action numbers for each state in value_function.

    Parameters
    ----------
    env: gym.core.Environment
      Environment to compute policy for. Must have nS, nA, and P as
      attributes.
    gamma: float
      Discount factor. Number in range [0, 1)
    value_function: np.ndarray
      Value of each state.

    Returns
    -------
    np.ndarray
      An array of integers. Each integer is the optimal action to take
      in that state according to the environment dynamics and the
      given value function.
    """    

    Q = np.zeros((env.nS, env.nA))
    for s in range(env.nS):
      for a in range(env.nA):
        for prob, nextstate, reward, is_terminal in env.P[s][a]:
          Q[s,a] += prob*(reward+gamma*value_func[nextstate])

    policy = Q.argmax(axis=-1).astype(int)

    return policy


def improve_policy(env, gamma, value_func, policy):
    """Given a policy and value function improve the policy.

    See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

        Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    value_func: np.ndarray
      Value function for the given policy.
    policy: dict or np.array
      The policy to improve. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    bool, np.ndarray
      Returns true if policy changed. Also returns the new policy.
    """


    newP = value_function_to_policy(env, gamma, value_func)
    policy_changed = np.any(policy != newP)

    return policy_changed, newP


def policy_iteration(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs policy iteration.

    See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

    You should use the improve_policy and evaluate_policy methods to
    implement this method.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    policy = np.zeros(env.nS, dtype='int')
    value_func = np.zeros(env.nS)

    for i in range(max_iterations):
      value_func = evaluate_policy(env, gamma, policy)
      policy_changed, policy = improve_policy(env, gamma, value_func, policy)

      if not policy_changed:
        break

    return policy, value_func, i, i


def value_iteration(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.

    See page 90 (pg 108 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """

    # initialize
    V = np.zeros(env.nS)

    # value iteration
    diff = 0.0
    for i in range(max_iterations):

      preV = V.copy()

      # calculate new Q and V value
      Q = np.zeros((env.nS, env.nA))
      for s in range(env.nS):
        for a in range(env.nA):
          for prob, nextstate, reward, is_terminal in env.P[s][a]:
            Q[s,a] += prob*(reward+gamma*V[nextstate])
      V = np.max(Q, axis=-1)

      # terminate condition check
      diff = np.max(np.abs(preV-V))
      if diff < tol:
        break

    # calculate policy
    policy = value_function_to_policy(env, gamma, V)

    return policy, i


def print_policy(policy, action_names):
    """Print the policy in human-readable format.

    Parameters
    ----------
    policy: np.ndarray
      Array of state to action number mappings
    action_names: dict
      Mapping of action numbers to characters representing the action.
    """
    str_policy = policy.astype('str')
    for action_num, action_name in action_names.items():
        np.place(str_policy, policy == action_num, action_name)

    print(str_policy)
