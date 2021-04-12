import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.input_data as input_data

#读取数据
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#需要的函数
def weight_varible(shape):  #权重随机初始化
	init=tf.random.truncated_normal(shape,stddev=0.1)
	return tf.Variable(init)
def bias_varible(shape):    #偏置随机初始化
	init=tf.constant(0.1,shape=shape)
	return tf.Variable(init)
def conv(x,y):  #卷积函数
	return tf.nn.conv2d(x,y,strides=[1,1,1,1],padding='SAME')
def max_pool(x):    #池化函数
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#创建一个会话
sess=tf.InteractiveSession()

tf.disable_eager_execution()
x=tf.placeholder(tf.float32,shape=[None,784])   #设置占位符，最后传入具体值再运算
y=tf.placeholder(tf.float32,shape=[None,10])
keep_prob=tf.placeholder(tf.float32)    #神经元个数
x_image=tf.reshape(x,[-1,28,28,1])

#卷积和池化
w_conv=weight_varible([5,5,1,16])  #5*5的卷积核，输入通道为1，输出通道为16
b_conv=bias_varible([16])   #偏置的数值
h_conv=tf.nn.relu(conv(x_image, w_conv) + b_conv)   #经过激活函数的输出
h_pool=max_pool(h_conv)   #通过池化层

#全连接
w_fc=weight_varible([14*14*16,10])    #转化为0-9的识别结果
b_fc=bias_varible([10])
h_pool2_flat=tf.reshape(h_pool,[-1,14*14*16])    #经过一次池化后图像变成14*14*16
h_fc=tf.matmul(h_pool2_flat,w_fc)+b_fc      #预测值

#梯度下降训练模型
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=h_fc))    #误差
train_step=tf.train.AdamOptimizer(1e-4).minimize(loss)  #梯度下降法
correct=tf.equal(tf.argmax(h_fc,1), tf.argmax(y, 1))
accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))    #将布尔类型转为数字0和1

sto=[]
sess.run(tf.global_variables_initializer())
for i in range(1,1001):
	batch = mnist.train.next_batch(50)  #一批50条数据
	if i % 100 == 0:
		train_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
		sto.append(train_accuracy)
		print("第", i, "批数据训练后的正确率", train_accuracy)
	train_step.run(feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})
plt.plot(sto)
plt.show()