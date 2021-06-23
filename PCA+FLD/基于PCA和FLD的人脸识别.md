

## 											基于PCA和FLD的人脸识别

一、实验目的

​		掌握人脸识别的基本方法，了解PCA和FLD的原理。



二、实验原理

​		一般的人脸识别包括图像预处理、特征提取、识别等步骤，本实验由三个函数分别实现这三步。

​		1、CreateDatabase函数实现图像预处理，即从文件路径中提取出图像数据，将二维图片转化成一维。

​		2、FisherfaceCore函数用于提取特征，它首先使用PCA方法，将预处理得到的灰度矩阵进行均值、中心化操作得到协方差矩阵L，再求出L的特征值和特征向量，最后结合背离度选出需要的特征向量。

​		PCA方法的缺点是需要处理的数据量较大，若用于大型的人脸识别工程可能效率不高，因此FisherfaceCore使用了FLD方法进行改进。FLD方法将PCA得到的特征向量投影到Fisher线性子空间，起到数据降维的作用，这样得到的特征向量数量级更少。

​		3、Recognition函数利用之前的特征向量进行判别，判别方法是计算欧式距离，取最小者即是最接近的判别结果。

​		本实验使用的ORL Faces数据库，包含40个不同人的400张图片。



三、实验步骤  

​		1、获得图像的特征值和特征向量，用matlab自带函数即可

```matlab
m_database = mean(T,2);          %训练样本矩阵的行均值
A = T - repmat(m_database,1,Pic);%背离矩阵A（M*N x Pic），计算每一张训练集图片相对于平均值的背离程度
L = A'*A;               % 协方差矩阵（Pic x Pic）
[V,D] = eig(L);         % V：特征向量矩阵  D：特征值对角阵
```

​		

​		2、投影到Fisher空间需要计算一些描述量，如各类样本均值向量、离散度矩阵：

```matlab
m_PCA = mean(ProjectedImages_PCA,2);         % 特征空间的总样本均值（Pic-C x 1）  
m_Intra = zeros(Pic-Class_number,Class_number);       %类内样本均值矩阵（Pic-C x C）
S_Within = zeros(Pic-Class_number,Pic-Class_number);  %类内离散度矩阵（Pic-C x Pic-C） 
S_Between = zeros(Pic-Class_number,Pic-Class_number);   %类间离散度矩阵（Pic-C x Pic-C）
```

![1](C:\Users\magic\OneDrive\桌面\1.PNG)

![2](C:\Users\magic\OneDrive\桌面\2.PNG)

​			类内离散度矩阵和类间离散度矩阵，它们表示样本到某向量投影点的离散程度

```matlab
for i = 1 : Class_number
    m_Intra(:,i) = mean( ( ProjectedImages_PCA(:,((i-1)*Class_population+1):i*Class_population) ), 2 )';   %每一类（人）的类内样本均值
    
    S  = zeros(Pic-Class_number,Pic-Class_number); 
    for j = ( (i-1)*Class_population+1 ) : ( i*Class_population )
        S = S + (ProjectedImages_PCA(:,j)-m_Intra(:,i))*(ProjectedImages_PCA(:,j)-m_Intra(:,i))';
    end
    
    S_Within = S_Within + S; 
    S_Between = S_Between + (m_PCA-m_Intra(:,i)) * (m_PCA-m_Intra(:,i))'; 
end    
```

​		

​		3、定义代价函数J，图像识别标准是类内样本投影尽可能密集，类间样本投影尽可能分散，即最大化类间离散度矩阵和最小化类内离散度矩阵。

![3](C:\Users\magic\OneDrive\桌面\3.PNG)

```matlab
[J_eig_vec, J_eig_val] = eig(S_Between,S_Within); %J_eig_val：广义特征值矩阵 J_eig_vec：广义特征向量矩阵 J = inv(S_Within) * S_Between
J_eig_vec = fliplr(J_eig_vec);                  %对向量矩阵进行列翻转
```

​		

​		4、寻找和样本向量欧氏距离最接近的测试图像

```matlab
Euc_dist = [];
for i = 1 : Train_Number
    q = ProjectedImages_Fisher(:,i);
    temp = ( norm( ProjectedTestImage - q ) )^2;
    Euc_dist = [Euc_dist temp]; 
end
```



四、实验结果

​		1、



​		![4](C:\Users\magic\OneDrive\桌面\4.PNG)

​		运行PCA_FLD，训练文件和测试文件选择同一个，如s1

![5](C:\Users\magic\OneDrive\桌面\5.PNG)

​		选择1-10中的一个图像进行测试，如3



![6](C:\Users\magic\OneDrive\桌面\6.PNG)

​		运行结果如上，测试图像和识别图像一致，识别成功。

![7](C:\Users\magic\OneDrive\桌面\7.PNG)

​		显示结果为3



​		2、再测试一个错误例子：训练文件选择s1，测试文件选择s3

![8](C:\Users\magic\OneDrive\桌面\8.PNG)

​		找到s3中最接近的图像，但显然不对。



​		3、可以结合背离度说明：正确识别的背离度起伏较小，能找到一个背离度近似为0的结果；而错误识别的背离度起伏很大，说明测试图像与训练图像差别较大，难以找出相似的结果。

![a](C:\Users\magic\OneDrive\桌面\a.jpg)

![b](C:\Users\magic\OneDrive\桌面\b.jpg)



五、实验心得

​		通过本次实验对人脸识别的常用方法有了基本了解。PCA的识别思想是提取图像的主要特点（特征向量），PCA寻找n个相互正交的向量，并用它们表征完整图像，减少了数据量。另外，PCA将图像当做一个整体来处理，不关心具体的五官、毛发的因素，简化了识别过程。

​		在PCA的基础上，FLD将样本投影到Fisher空间 ，进一步减少了数据量。由于本实验只训练10张图像并识别一张图像，数据较少所以难以察觉速度差别。