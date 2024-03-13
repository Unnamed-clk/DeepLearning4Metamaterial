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

x_data = data_raw[1]
x_data = np.delete(x_data,[0,1,2,3,4],axis=0) # type: ignore

y_data = data_raw[0]
y_data = np.delete(y_data,[0,1,2,3,4],axis=0) # type: ignore

x_data = tf.expand_dims(x_data,-1)

neurons = 650
l2_w = 0.015
l2_b = 0.015
alpha = 0.05
epochs = 500
batch_size = 32
learning_rate = 0.001
shape = (201,4)
loss_curve = np.zeros((epochs,1))

all_dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))

test_dataset = all_dataset.take(300)
train_dataset = all_dataset.skip(300)

test_dataset.save('/test')

train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
test_dataset = test_dataset.shuffle(buffer_size=1024).batch(batch_size)

class CNN(Model):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.conv1 = layers.Conv2D(32,(4,1),input_shape=(201,4,1))
		self.max1 = layers.MaxPool2D((3,1))
		self.conv2 = layers.Conv2D(64,(3,1),activation='relu',input_shape=(66,4,1))
		self.max2 = layers.MaxPool2D((4,1))
		self.conv3 = layers.Conv2D(128,(9,1),activation='relu',input_shape=(16,4,1))
		self.max3 = layers.MaxPool2D((2,1))
		self.conv4 = layers.Conv2D(256,(4,2),activation='relu',input_shape=(4,4,1))
		self.flat = layers.Flatten()
		self.dense1 = layers.Dense(512,activation='relu')
		self.dense2 = layers.Dense(128,activation='relu')
		self.dense3 = layers.Dense(4,activation='relu')
	def call(self, inputs, training=None, mask=None):
		x = self.conv1(inputs)
		x = self.max1(x)
		x = self.conv2(x)
		x = self.max2(x)
		x = self.conv3(x)
		x = self.max3(x)
		x = self.conv4(x)
		x = self.flat(x)
		x = self.dense1(x)
		x = self.dense2(x)
		outputs = self.dense3(x)
		return outputs

cnn = CNN()

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss_fn = tf.keras.losses.MeanSquaredError()

train_acc_metric = keras.metrics.MeanAbsoluteError()
test_acc_metric = keras.metrics.MeanAbsoluteError()

train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)

cnn.compile(loss='mse',
			optimizer='adam')

@tf.function
def train_step(x,y):
	with tf.GradientTape() as tape:
		y_pred = cnn(x,training=True)
		loss_value = loss_fn(y_true=y,y_pred=y_pred)
	grads = tape.gradient(loss_value,cnn.trainable_weights)
	optimizer.apply_gradients(zip(grads, cnn.trainable_weights))
	train_acc_metric.update_state(y_true=y, y_pred=y_pred)
	return loss_value

@tf.function
def test_step(x,y):
	y_val = cnn(x,training=False)
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
	loss_curve[epoch] = float(loss_value) # type: ignore
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

cnn.save('cnn',save_format="tf")

np.savetxt('loss_curve',loss_curve)

	

