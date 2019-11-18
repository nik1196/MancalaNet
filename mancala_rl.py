import tensorflow as tf, mancala, numpy as np, random
from collections import deque
import matplotlib.pyplot as plt


game = mancala.Mancala()

model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation = tf.nn.relu, input_shape=(14,)),
    tf.keras.layers.Dense(16, activation = tf.nn.relu),
    tf.keras.layers.Dense(16, activation = tf.nn.relu),
    tf.keras.layers.Dense(14)
    ])



learning_rate = 1e-3
optimizer = tf.keras.optimizers.Adam(learning_rate)

model.compile(loss=tf.keras.losses.mean_squared_error.__name__, optimizer=optimizer)


num_episodes = 1000
num_turns_per_episode = 500

epsilon = 1.0
epsilon_min = 0.5
epsilon_decay = 0.995

gamma = 0.95


fig = plt.figure()
p1_plot = fig.add_subplot(211)
p2_plot = fig.add_subplot(212)
fig_x = []
fig_y1 = []
fig_y2 = []


for episode in range(num_episodes):
    game_turns = [deque(maxlen=2000), deque(maxlen=2000)]
    wrong_pit_penalty = 1e-2
    game.reset()
    cur_state = game.get_pits()
    cur_state = np.reshape(cur_state, [1,14])
    turn = 0.0
    done = False
    scores = np.zeros([1,2])[0]
    while not done:
        turn += 0.5
        player = (turn/0.5)%2
        print("Turn: ",turn, "Player: ", player)
        best_action = 0
        print(cur_state)
        penalties = model.predict(cur_state)[0]
        for j in range(7,14):
            penalties[j] = wrong_pit_penalty
        if np.random.rand() >= epsilon:
            best_action = random.randrange(7)
        else:
            rewards = model.predict(cur_state)[0]
            choice = np.argmax(rewards)
            print(choice)
            print(cur_state[0][choice])
            while choice > 6 or cur_state[0][choice] == 0:
                penalties[choice] *= 0.1
                if penalties[choice] < 1e-3:
                    for i in range(len(penalties)):
                        if i != choice:
                            penalties[i] += 1
                model.fit(cur_state,np.reshape(penalties, [1,14]), epochs=1,verbose=0)
                rewards = model.predict(cur_state)[0]
                choice = np.argmax(rewards)
                penalties[:7] = rewards[:7]
##                print(penalties[0][choice], choice)
            print("Chose right pit: ", choice)
            best_action = choice
        next_state, score = game.play(best_action, return_state = True)
        print(next_state)
        done = game.done()
        game_turns[int(player)].append((cur_state, best_action, np.reshape(next_state,[1,14]), score, done))
        cur_state = next_state[::-1]
        game.set_pits(cur_state)
        cur_state = np.reshape(cur_state, [1,14])
        scores[int(player)] += score
        if done == True:
            print("Episode:", episode+1, " Scores:", scores, " Epsilon:", epsilon)
            fig_x.append(episode)
            fig_y1.append(scores[0])
            fig_y2.append(scores[1])
            break
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        else:
            epsilon = 1.0
    print(len(game_turns[0]))
    for i in range(len(game_turns)):
        lose_penalty = int(scores[i] < scores[(i+1)%2])*0.1
        for cur_state, best_action, next_state, score, done in game_turns[i]:
            potential_reward = score
            if done == False:
                potential_reward = score + gamma*np.amax(model.predict(next_state)[0])*(-lose_penalty)
            target_reward = model.predict(cur_state)
            target_reward[0][best_action] = potential_reward
            model.fit(cur_state,target_reward, epochs=1,verbose=0)

p1_plot.plot(fig_x,fig_y1)
p2_plot.plot(fig_x,fig_y2)
plt.show()
game.reset()
cur_state = game.get_pits()
cur_state = np.reshape(cur_state, [1,14])
game_over = False
player = 0
scores = np.zeros([1,2])[0]
while not game_over:
    print("Pit state: ", game.get_pits())
    action = np.argmax(model.predict(cur_state)[0])
    next_state, reward = game.play(action, return_state = True)
    cur_state = next_state[::-1]
    print("Pit state: ", game.get_pits())
    game.set_pits(cur_state)
    cur_state = np.reshape(cur_state, [1,14])
    game_over = game.done()
    print("Player ",player + 1, " played:", action)

    scores[player] += reward
    player = (player + 1) % 2
print("Winner: Player ", player + 1)
print("Scores: ", scores)
