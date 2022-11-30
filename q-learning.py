import random
import matplotlib.pyplot as plt

WORLD_HEIGHT = 7
WORLD_WIDTH = 10
WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
START = [3, 0]
GOAL = [3, 7]
ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]
Q = {(i, j):[0, 0, 0, 0] for i in range(7) for j in range(10)}
EPSILON = 1
LAMBDA = 1
GAMA = 0.9
TOTAL_REWARD = []


def step(state, action):
    i, j = state
    if action == ACTION_UP:
        state = [max(i - 1 - WIND[j], 0), j]
    elif action == ACTION_DOWN:
        state = [max(min(i + 1 - WIND[j], WORLD_HEIGHT - 1), 0), j]
    elif action == ACTION_LEFT:
        state = [max(i - WIND[j], 0), max(j - 1, 0)]
    elif action == ACTION_RIGHT:
        state = [max(i - WIND[j], 0), min(j + 1, WORLD_WIDTH - 1)]
    reward = -1.0
    if state == GOAL:
        reward = 50
    return state, reward


for i in range(10000):
    CURRENT_STATE = START
    goal_founded = False
    GAME_REWARD = []
    for j in range(50):
        if random.random() < EPSILON:
            ACTION = random.choice(ACTIONS)
        else:
            max_val = max(Q[tuple(CURRENT_STATE)])
            ACTION = Q[tuple(CURRENT_STATE)].index(max_val)
        NEXT_STATE, REWARD = step(CURRENT_STATE, ACTION)
        GAME_REWARD.append(REWARD)
        value = max(Q[tuple(NEXT_STATE)])
        Q[tuple(CURRENT_STATE)][ACTION] = LAMBDA * (REWARD + GAMA * value) + (1 - LAMBDA) * Q[tuple(CURRENT_STATE)][ACTION]
        if REWARD == 50:
            LAMBDA *= 0.99
            EPSILON *= 0.98
            goal_founded = True
            print(f'episode {j}: goal has been found in step {j}')
            break
        CURRENT_STATE = NEXT_STATE
    TOTAL_REWARD.append(sum(GAME_REWARD))

plt.plot(TOTAL_REWARD)
plt.show()
