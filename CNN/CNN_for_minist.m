clear 
clc 
close all
load('MNIST_for_BP.mat')
tic
% train_ima = loadMNISTImages(url);
% train_lab = loadMNISTLabels(url);
% test_ima  = loadMNISTImages(url); %data after reshape , original size is 28x28
% test_lab  = loadMNISTLabels(url); %label
% 读取四个文件
%-----------Training Part-----------%
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
sto=[];

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
        sto=[sto,correctRate];
        disp(['第',num2str(num),'次训练正确率为：',num2str(correctRate)])
    end
end
plot(sto)
toc