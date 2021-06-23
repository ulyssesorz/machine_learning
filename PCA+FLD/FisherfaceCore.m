function [m_database,V_PCA,ProjectedImages_PCA] = FisherfaceCore(T)
%m_database：训练样本矩阵的行均值  V_PCA：PCA空间传递矩阵  
%V_Fisher：向量矩阵J中最大的C-1个特征向量  ProjectedImages_Fisher：fisher线性子空间
        

%训练集图片z1
Class_number = ( size(T,2) )/2;                  %类数（即人数）
Class_population = 2;                               %每一类（每个人）的图片数
Pic = Class_population * Class_number;      %总图片数

m_database = mean(T,2);                         %训练样本矩阵的行均值

A = T - repmat(m_database,1,Pic);        %背离矩阵A（M*N x Pic），计算每一张训练集图片相对于平均值的背离程度
L = A'*A;                 % 协方差矩阵（Pic x Pic）
[V,D] = eig(L);         % V：特征向量矩阵  D：特征值对角阵


%过滤掉小的特征值对应的特征向量
L_eig_vec = [];
for i = 1 : Pic-Class_number 
    L_eig_vec = [L_eig_vec V(:,i)];
end

%V_PCA：PCA空间传递矩阵（M*N x Pic-C）  
V_PCA =   A * L_eig_vec; 

%投影到PCA特征空间
ProjectedImages_PCA = [];
for i = 1 : Pic
    temp = V_PCA'*A(:,i);   
    ProjectedImages_PCA = [ProjectedImages_PCA temp];  %PCA特征子空间（Pic-C x Pic）
end