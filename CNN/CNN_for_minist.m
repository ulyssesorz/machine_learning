clear 
clc 
close all
load('MNIST_for_BP.mat')
tic
% train_ima = loadMNISTImages(url);
% train_lab = loadMNISTLabels(url);
% test_ima  = loadMNISTImages(url); %data after reshape , original size is 28x28
% test_lab  = loadMNISTLabels(url); %label
% ��ȡ�ĸ��ļ�
%-----------Training Part-----------%
%����������
sam_sum = 60000;    %ѵ����������
input = 784;        %�������Ԫ��Ŀ��28*28��ͼƬ��
output = 10;        %���ʶ��Ľ��(0-9)
hid = 20;           %���ز���Ԫ��Ŀ
w1 = randn([input,hid]);%��ʼ������ˣ�ͬһ�����ʵ��Ȩֵ����
Temporary_variable = w1;
w2 = randn([hid,output]);
bias1 = zeros(hid,1);   %ƫ�õ�Ԫ
bias2 = zeros(output,1);
rate1 = 0.005;
rate2 = 0.005;          %����ѧϰ��

%�����������������ʱ�洢����
temp1 = zeros(hid,1);
net = temp1;
temp2 = zeros(output,1);
z = temp2;
sto=[];

for num = 1:100		%ѵ������
for i = 1:sam_sum	%ѵ��������   
    label = zeros(10,1);	%����������
    label(train_lab(i)+1,1) = 1;

    %ǰ���㷨
    %�˴�ѡ��784*1��train_ima���������Ҫ��(:,1)
    temp1 = train_ima(:,i)'*w1 + bias1';%��һ������
    net = sigmoid(temp1);%��һ�㾭������������
    %�˴�ѡ��hid*1�����ز����
    temp2 = net*w2 + bias2';%�ڶ�������
    z = sigmoid(temp2);%�ڶ��㾭������������
    z = z';net = net';
    
    %bp�㷨 ���¾����
    error = label - z;%��������
    deltaZ = error.*z.*(1-z);%��������������ƫ������ z.*(1-z)��sigmoid�ĵ���
    deltaNet = net.*(1-net).*(w2*deltaZ);%�������ز������ƫ��������ʽ����
    for j = 1:output
        w2(:,j) = w2(:,j) + rate2*deltaZ(j).*net;%�ݶ��½�������w2��
    end
    for j = 1:hid
        w1(:,j) = w1(:,j) + rate1*deltaNet(j).*train_ima(:,i);%����w1
    end
    bias2 = bias2 + rate2*deltaZ;%����ƫ��
    bias1 = bias1 + rate1*deltaNet;
end
%���Բ���
test_sum = 10000;
count = 0;
for i = 1:test_sum
    temp1 = test_ima(:,i)'*w1 + bias1';
    net = sigmoid(temp1);
    %�˴�ѡ��hid*1�����ز����
    temp2 = net*w2 + bias2';
    z = sigmoid(temp2);
    
    [maxn,inx] = max(z);    %�õ������������
    inx = inx -1;   %������
    if inx == test_lab(i)
        count = count+1;
    end
end
    if(mod(num,10)==0)
        correctRate=count/test_sum;
        sto=[sto,correctRate];
        disp(['��',num2str(num),'��ѵ����ȷ��Ϊ��',num2str(correctRate)])
    end
end
plot(sto)
toc