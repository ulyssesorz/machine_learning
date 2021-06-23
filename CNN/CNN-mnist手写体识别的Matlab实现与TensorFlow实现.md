

## 											**CNN-mnist手写体识别的Matlab实现与TensorFlow实现**

一、实验目的

​		使用卷积神经网络对mnist数据集的手写体数据进行分类识别。

二、实验原理

​		CNN的局部感受野、权值共享、池化等特点能将大量的数据进行降维，训练速度快，能捕捉数据的局部特征进行分类，很适合用于图像识别。

​		使用CNN进行识别分类时，图像以像素矩阵的形式输入。首先，卷积层将这个较大的矩阵映射为一个较小的矩阵，该过程提取了图片每个小部分的特征，卷积而非全连接使得数据量大大减少。接着池化层对卷积层的输出进行下采样，再次对数据降维，然后全连接层将特征汇总到一起，转化为十位的输出（0-9），即得到所需的数字识别结果。

三、实验步骤

​		Matlab实现：

​		1、初始化参数。主要是每层神经元的数值以及权值

```matlab
%三层神经网络
sam_sum = 60000;    %训练样本数量
input = 784;        %输入的神经元数目（28*28的图片）
output = 10;        %输出识别的结果(0-9)
hid = 20;           %隐藏层神经元数目
w1 = randn([input,hid]);%初始化卷积核，同一卷积核实现权值共享
Temporary_variable = w1;
w2 = randn([hid,output]);
bias1 = zeros(hid,1);   %偏置单元
bias2 = zeros(output,1);
rate1 = 0.005;
rate2 = 0.005;          %设置学习率

%用作三层神经网络的暂时存储变量
temp1 = zeros(hid,1);
net = temp1;
temp2 = zeros(output,1);
z = temp2;
```

​		2、训练部分。先进行一次前向传播得到初始参数，再使用bp算法更新权值，训练100次。

​			（此处对原本的代码做了修改：将训练部分和测试部分写在一起，这样可以动态观察训练正确率的变化）

```matlab
for num = 1:100		%训练次数
for i = 1:sam_sum	%训练集总数   
    label = zeros(10,1);	%储存输出结果
    label(train_lab(i)+1,1) = 1;

    %前向算法
    %此处选用784*1的train_ima矩阵，因此需要加(:,1)
    temp1 = train_ima(:,i)'*w1 + bias1';%第一层的输出
    net = sigmoid(temp1);%第一层经过激活函数的输出
    %此处选用hid*1的隐藏层矩阵
    temp2 = net*w2 + bias2';%第二层的输出
    z = sigmoid(temp2);%第二层经过激活函数的输出
    z = z';net = net';
    
    %bp算法 更新卷积核
    error = label - z;%输出层误差
    deltaZ = error.*z.*(1-z);%计算输出层参数的偏导数， z.*(1-z)是sigmoid的导数
    deltaNet = net.*(1-net).*(w2*deltaZ);%计算隐藏层参数的偏导数（链式法则）
    for j = 1:output
        w2(:,j) = w2(:,j) + rate2*deltaZ(j).*net;%梯度下降法更新w2，
    end
    for j = 1:hid
        w1(:,j) = w1(:,j) + rate1*deltaNet(j).*train_ima(:,i);%更新w1
    end
    bias2 = bias2 + rate2*deltaZ;%更新偏置
    bias1 = bias1 + rate1*deltaNet;
end

%测试部分
test_sum = 10000;
count = 0;
for i = 1:test_sum
    temp1 = test_ima(:,i)'*w1 + bias1';
    net = sigmoid(temp1);
    %此处选用hid*1的隐藏层矩阵
    temp2 = net*w2 + bias2';
    z = sigmoid(temp2);
    
    [maxn,inx] = max(z);    %得到概率最大的输出
    inx = inx -1;   %输出结果
    if inx == test_lab(i)
        count = count+1;
    end
end
    if(mod(num,10)==0)
        correctRate=count/test_sum;
        disp(['第',num2str(num),'次训练正确率为：',num2str(correctRate)])
    end
end
```



​		Tensorflow实现：

​		1、导入需要的库，读入数据，编写功能函数。

```python
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
```

​		2、训练模型。卷积、池化、全连接各一层，使用梯度下降法优化模型。

```python
#卷积层和池化层
w_conv=weight_varible([5,5,1,16])  #5*5的卷积核，输入通道为1，输出通道为16
b_conv=bias_varible([16])   #偏置的数值
h_conv=tf.nn.relu(conv(x_image, w_conv) + b_conv)   #经过激活函数的输出
h_pool=max_pool(h_conv)   #通过池化层

#全连接层
w_fc=weight_varible([14*14*16,10])    #转化为0-9的识别结果
b_fc=bias_varible([10])
h_pool2_flat=tf.reshape(h_pool,[-1,14*14*16])    #经过一次池化后图像变成14*14*16
h_fc=tf.matmul(h_pool2_flat,w_fc)+b_fc      #预测值

#梯度下降训练模型
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=h_fc))    #误差
train_step=tf.train.AdamOptimizer(1e-4).minimize(loss)  #梯度下降法
correct=tf.equal(tf.argmax(h_fc,1), tf.argmax(y, 1))
accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))    #将布尔类型转为数字0和1
```

​		3、测试部分。批处理，使用feed_dict方式给模型喂数据

```python
sess.run(tf.global_variables_initializer())
for i in range(1,1001):
	batch = mnist.train.next_batch(50)  #一批50条数据
	if i % 100 == 0:
		train_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
		print("第", i, "批数据训练后的正确率", train_accuracy)
	train_step.run(feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})
```

四、实验结果

​		Matlab实现：正确率从75%提升到89%

![2](C:\Users\magic\OneDrive\桌面\2.PNG)

![untitled](C:\Users\magic\OneDrive\桌面\untitled.jpg)

​		Tensorflow实现：结果与上面类似，正确率整体呈上升趋势，最终接近90%

​		![1](C:\Users\magic\OneDrive\桌面\1.PNG)

![Figure_1](C:\Users\magic\OneDrive\桌面\Figure_1.png)



五、实验心得

​		掌握CNN的理论知识后，成功编程实现了一个用于手写体识别的卷积神经网络。经过本次实验，对CNN的思想理解加深了很多，明白了图像从卷积到判别的具体过程和原理。

​		Matlab的实现很好地还原了卷积神经网络的数学原理，尤其是训练时BP算法的链式法则通过代码能清晰地表现出来。

​		TensorFlow的实现则更为简单易用，丰富的第三方库屏蔽了很多数学推导，如卷积函数和梯度下降都可以直接用内置函数实现。整个过程更为直观，代码量也较少。

