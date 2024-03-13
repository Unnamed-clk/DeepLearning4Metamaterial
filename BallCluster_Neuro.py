import tensorflow as tf
import numpy as np
import os
import BallCluster_getdata as getdata
import datetime

from tensorflow import keras 
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import layers

datapath = ['ForDL']
paras = ['r','l','d','c']
neurons = len(os.listdir(datapath[0]))

if datapath[0]+'.npz' in os.listdir('.'):
	data_npz = np.load(datapath[0]+'.npz')
	data_raw = (data_npz['para'],data_npz['S'],data_npz['F'])
else:
	data_raw = getdata.getpara(datapath[0],paras,201,'s2p',normal=True)

x_data = data_raw[0]
y_data = data_raw[1]

neurons = 650
l2_w = 0.015
l2_b = 0.015
alpha = 0.05
epochs = 500
batch_size = 32
learning_rate = 0.001
shape = (201,2)
loss_curve = np.zeros((epochs,2))

all_dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))

test_dataset = all_dataset.take(100)
train_dataset = all_dataset.skip(100)

test_dataset.save('/test')

train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
test_dataset = test_dataset.shuffle(buffer_size=1024).batch(batch_size)

class FNN(Model):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.dense0 = layers.Dense(neurons, input_dim = 4, use_bias =True, kernel_initializer = 'glorot_normal', bias_initializer = 'glorot_normal',kernel_regularizer=tf.keras.regularizers.l2(l2_w), bias_regularizer=tf.keras.regularizers.l2(l2_b))
		self.dense1 = layers.Dense(neurons, use_bias =True, kernel_initializer = 'glorot_normal', bias_initializer = 'glorot_normal',kernel_regularizer=tf.keras.regularizers.l2(l2_w), bias_regularizer=tf.keras.regularizers.l2(l2_b))
		self.leakyrelu = layers.LeakyReLU(alpha=alpha)
		self.dense2 = layers.Dense(804, use_bias =True, kernel_initializer = 'glorot_normal', bias_initializer = 'glorot_normal',kernel_regularizer=tf.keras.regularizers.l2(l2_w), bias_regularizer=tf.keras.regularizers.l2(l2_b))
		self.reshape = layers.Reshape((201,4))
	def call(self, inputs, training=None, mask=None):
		x = self.dense0(inputs)
		x = self.dense1(x)
		x = self.leakyrelu(x)
		x = self.dense2(x)
		output = self.reshape(x)
		return output

fnn = FNN()

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss_fn = tf.keras.losses.MeanSquaredError()

train_acc_metric = keras.metrics.MeanSquaredError()
test_acc_metric = keras.metrics.MeanSquaredError()

train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)

fnn.compile(loss='mse',
			optimizer='adam')

@tf.function
def train_step(x,y):
	with tf.GradientTape() as tape:
		y_pred = fnn(x,training=True)
		loss_value = loss_fn(y_true=y,y_pred=y_pred)
	grads = tape.gradient(loss_value,fnn.trainable_weights)
	optimizer.apply_gradients(zip(grads, fnn.trainable_weights))
	train_acc_metric.update_state(y_true=y, y_pred=y_pred)
	return loss_value

@tf.function
def test_step(x,y):
	y_val = fnn(x,training=False)
	test_acc_metric.update_state(y,y_val)

bar = getdata.ProgressBar(neurons/batch_size,fmt=getdata.ProgressBar.FULL)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'log/gradient_tape/' + current_time + '/train'
test_log_dir = 'log/gradient_tape/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir) # type: ignore
test_summary_writer = tf.summary.create_file_writer(test_log_dir) # type: ignore

for epoch in range(epochs):
	# print('No.{} epoch:'.format(epoch))
	for x_batch_train,y_batch_train in train_dataset:
		loss_value = train_step(x_batch_train,y_batch_train)
		bar()
		bar.current += 1
	with train_summary_writer.as_default():
		tf.summary.scalar('loss', train_loss.result(), step=epoch)
		tf.summary.scalar('accuracy', train_acc_metric.result(), step=epoch)
	bar.done()
	print(float(loss_value)) # type: ignore
	loss_curve[epoch,1] = float(loss_value) # type: ignore
	loss_curve[epoch,0] = epoch
	for x_batch_val, y_batch_val in test_dataset:
		test_step(x_batch_val, y_batch_val)
	with test_summary_writer.as_default():
		tf.summary.scalar('loss', test_loss.result(), step=epoch)
		tf.summary.scalar('accuracy', test_acc_metric.result(), step=epoch)

	template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
	print (template.format(epoch+1,
							train_loss.result()*100, 
							train_acc_metric.result()*100,
							test_loss.result()*100, 
							test_acc_metric.result()*100))
	 # Reset metrics every epoch
	train_loss.reset_states()
	test_loss.reset_states()
	train_acc_metric.reset_states()
	test_acc_metric.reset_states()

fnn.save('FNN',save_format=tf)

fnn.summary()

np.savetxt('fnn_loss.txt',loss_curve)