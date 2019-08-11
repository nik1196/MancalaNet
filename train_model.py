import time, tensorflow as tf, numpy as np, matplotlib.pyplot as plt, mancala

game = mancala.Mancala()
model = tf.keras.Sequential([
    tf.keras.layers.Dense(28, activation = tf.nn.relu, input_shape=(14,)),
    tf.keras.layers.Dense(14, activation = tf.nn.softmax)
     ])

optimizer = tf.keras.optimizers.SGD(0.01)

model.compile(loss=tf.keras.losses.categorical_crossentropy.__name__, optimizer=optimizer,metrics=['accuracy'])
epochs = 200

def generate_data(size):
    ip = []
    op = []
    for i in range(size):
        ip.append(game.generate_state())
        op.append(game.get_best_pit())
    return np.array(ip, ndmin=2), np.array(op,ndmin=2)

data_size = 100000
ips, ops = generate_data(data_size)
history = model.fit(ips,ops,batch_size=data_size,epochs=epochs,validation_split=0.2,verbose=1)

model.save('new model', save_format='h5')
fig = plt.figure()
p1 = fig.add_subplot(121)
p2 = fig.add_subplot(122)
p2.set_ylim(0,1)
p2.grid()
p1.grid()
p2.set_yticks(np.arange(0,1,0.1))
x = [i for i in range(epochs)]
y = history.history['loss']
y2 = history.history['acc']
p1.plot(x,y, 'r', label='loss')
p1.legend()
p2.plot(x,y2, 'b', label='accuracy')
p2.legend()
plt.show()

