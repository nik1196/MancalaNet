import time, tensorflow as tf, numpy as np, matplotlib.pyplot as plt, mancala, random

game = mancala.Mancala()
model = tf.keras.Sequential([
    tf.keras.layers.Dense(624, activation = tf.nn.relu, input_shape=(14,)),
    tf.keras.layers.Dense(624, activation = tf.nn.relu),
    tf.keras.layers.Dense(14, activation = tf.nn.softmax)
     ])


learning_rate = 0.01
optimizer = tf.keras.optimizers.Adam(learning_rate)

model.compile(loss=tf.keras.losses.categorical_crossentropy.__name__, optimizer=optimizer,metrics=['accuracy'])
epochs = 1000

def generate_data(size):
    ip = []
    op = []
    for i in range(size):
        ip.append(game.generate_state())
        op.append(game.get_best_pit())
    return np.array(ip, ndmin=2), np.array(op,ndmin=2)

def hash_func(pits):
    hash_string = ''
    for i in range(len(pits)):
        hash_string += hex(pits[i])
    return hash_string

train_data_dict = {}
#play match with random choices, ignoring score
def generate_match_data(game,turns=9999):
    train_data = []
    train_labels = []
    pit_p1 = 0
    pit_p2 = 0
    random.seed()
    for turn in range(turns):
        pits = game.get_pits()
        if all(pit == 0 for pit in pits[:7]) or all(pit == 0 for pit in pits[7:]):
            break
        if hash_func(pits) not in train_data_dict:
            train_data.append(list(pits))
            pit_p2_ohvec = game.get_best_pit()
            train_labels.append(list(pit_p2_ohvec))
            train_data_dict[hash_func(pits)] = 1
        pit_p1 = random.randrange(1,8)
        game.play(pit_p1)
        if hash_func(pits) not in train_data_dict:
            train_data.append(list(pits))
            pit_p2_ohvec = game.get_best_pit()
            train_labels.append(list(pit_p2_ohvec))
            train_data_dict[hash_func(pits)] = 1
        pit_p2 = game.get_best_pit(False)
        game.play(pit_p2)
    game.reset()
    for turn in range(turns):
        pits = game.get_pits()
        if all(pit == 0 for pit in pits[:7]) or all(pit == 0 for pit in pits[7:]):
            break
        if hash_func(pits) not in train_data_dict:
            train_data.append(list(pits))
            pit_p2_ohvec = game.get_best_pit()
            train_labels.append(list(pit_p2_ohvec))
            train_data_dict[hash_func(pits)] = 1
        pit_p1 = random.randrange(1,8)
        game.play(pit_p1)
        if hash_func(pits) not in train_data_dict:
            train_data.append(list(pits))
            pit_p2_ohvec = game.get_best_pit()
            train_labels.append(list(pit_p2_ohvec))
            train_data_dict[hash_func(pits)] = 1
        pit_p2 = random.randrange(7,14)
        game.play(pit_p2)
    game.reset()
    for turn in range(turns):
        pits = game.get_pits()
        if all(pit == 0 for pit in pits[:7]) or all(pit == 0 for pit in pits[7:]):
            break
        if hash_func(pits) not in train_data_dict:
            train_data.append(list(pits))
            pit_p2_ohvec = game.get_best_pit()
            train_labels.append(list(pit_p2_ohvec))
            train_data_dict[hash_func(pits)] = 1
        pit_p1 = game.get_best_pit(False)
        game.play(pit_p1)
        if hash_func(pits) not in train_data_dict:
            train_data.append(list(pits))
            pit_p2_ohvec = game.get_best_pit()
            train_labels.append(list(pit_p2_ohvec))
            train_data_dict[hash_func(pits)] = 1
        pit_p2 = game.get_best_pit(False)
        game.play(pit_p2)
    return train_data, train_labels
    game.reset()

num_matches = 32000

ips = []
ops = []

print('Generating data:')
for match in range(num_matches):
    cur_ips, cur_ops = generate_match_data(game)
    ips.extend(list(cur_ips))
    ops.extend(list(cur_ops))
    
ips = np.array(ips, ndmin = 2)
ops = np.array(ops, ndmin = 2)
print(ips)
print(ops)

def scheduler(epoch):
    if epoch <= 500:
        return learning_rate
    else:
        return learning_rate / 10 #* (int((epoch-100)/100)))
##, callbacks=[tf.keras.callbacks.LearningRateScheduler(scheduler)]
history = model.fit(ips,ops,batch_size=int(len(ips)/2),epochs=epochs,validation_split=0.2,verbose=1)

model.save('new model2', save_format='h5')
fig = plt.figure()
p1 = fig.add_subplot(221)
p2 = fig.add_subplot(222)
p3 = fig.add_subplot(223)
p4 = fig.add_subplot(224)
p2.set_ylim(0,1)
p4.set_ylim(0,1)
p1.grid()
p2.grid()
p3.grid()
p4.grid()
p2.set_yticks(np.arange(0,1,0.1))
p4.set_yticks(np.arange(0,1,0.1))
x = [i for i in range(epochs)]
y = history.history['loss']
y2 = history.history['acc']
y3 = history.history['val_loss']
y4 = history.history['val_acc']
p1.plot(x,y, 'r', label='loss')
p1.legend()
p2.plot(x,y2, 'b', label='accuracy')
p2.legend()
p3.plot(x,y3, 'r', label='val_loss')
p3.legend()
p4.plot(x,y4, 'b', label='val_accuracy')
p4.legend()
plt.show()

