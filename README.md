# POLICY ITERATION ALGORITHM

### Name: DEEPIKA P
### Register Number: 212223240024

## AIM
Implement policy iteration algorithm to find optimal policy by iteratively maximizing the value function.

## PROBLEM STATEMENT
The objective of this experiment is to determine the optimal policy for a Markov Decision Process (MDP) using the Policy Iteration algorithm. Policy iteration consists of two main steps: policy evaluation and policy improvement. In the policy evaluation step, the value function for each state is calculated based on the current policy. In the policy improvement step, the action-value functions are compared to update and obtain the best possible policy for the MDP.
## POLICY ITERATION ALGORITHM
# Step 1:
Import required libraries.
# Step 2:
Load the frozen lake environment.
# Step 3:
Define the value evaluation, value improvement and value iteration functions.
# Step 4: 
Run the functions and display the results.</br>
</br>

## POLICY IMPROVEMENT FUNCTION

```python
def policy_improvement(V, P, gamma=1.0):
    Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
    for s in range(len(P)):
        for a in range(len(P[s])):
            for prob, next_state, reward, done in P[s][a]:
                Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))
    new_pi = lambda s: np.argmax(Q[s])
    return new_pi
```
## POLICY ITERATION FUNCTION

```python
def policy_iteration(P, gamma=1.0, theta=1e-10):
    pi = np.zeros(len(P), dtype=int)
    while True:
        pi_func = lambda s: pi[s]
        V = policy_evaluation(pi_func, P, gamma, theta)
        policy_stable = True
        for s in range(len(P)):
            old_action = pi[s]
            Q = np.zeros(len(P[s]))
            for a in range(len(P[s])):
                for prob, next_state, reward, done in P[s][a]:
                    Q[a] += prob * (reward + gamma * V[next_state] * (not done))
            pi[s] = np.argmax(Q)
            if old_action != pi[s]:
                policy_stable = False
        if policy_stable:
            break

    return V, lambda s: pi[s]
```

## OUTPUT:
### 1. Policy, Value function and success rate for the Adversarial Policy
</br>
<img width="626" height="412" alt="image" src="https://github.com/user-attachments/assets/d0f1ce2c-5160-4bea-a290-1b6a89ef9583" />


</br>

### 2. Policy, Value function and success rate for the Improved Policy
</br>
<img width="662" height="390" alt="image" src="https://github.com/user-attachments/assets/61c6da74-ebcd-4039-b85b-7496ad9e421c" />


</br>

### 3. Policy, Value function and success rate after policy iteration
</br>
<img width="625" height="404" alt="image" src="https://github.com/user-attachments/assets/c528dfd4-15b4-46a7-87d5-413fc59bbb67" />


</br>


## RESULT:

Therefore, policy iteration algorithm to find optimal policy by iteratively maximizing the value function is successfully implemented.
