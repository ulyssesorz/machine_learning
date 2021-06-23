clear all
clc
close all

%��ȡѵ�����Ͳ���ͼƬ
TrainDatabasePath = uigetdir(strcat(matlabroot,'\FLD'), 'Select training database path' );
TestDatabasePath = uigetdir(strcat(matlabroot,'\FLD'), 'Select test database path');

prompt = {'Enter test image name (a number between 1 to 10):'};
dlg_title = 'Input of FLD-Based Face Recognition System';
num_lines= 1;
def = {'1'};

TestImage  = inputdlg(prompt,dlg_title,num_lines,def);
TestImage = strcat(TestDatabasePath,'\',char(TestImage),'.pgm');
im = imread(TestImage);

%ѵ��������
T = CreateDatabase(TrainDatabasePath);
%PCA+FLD
[m_database,V_PCA,ProjectedImages_PCA] = FisherfaceCore(T);
%����б���ͼƬ
[OutputName,d] = Recognition(TestImage,m_database,V_PCA,ProjectedImages_PCA);

SelectedImage = strcat(TrainDatabasePath,'\',OutputName);
SelectedImage = imread(SelectedImage);

imshow(im)
title('Test Image');
figure,imshow(SelectedImage);
title('Equivalent Image');

str = strcat('Matched image is :  ',OutputName);
disp(str);
figure;plot(d);title('����ʶ��ı����');
