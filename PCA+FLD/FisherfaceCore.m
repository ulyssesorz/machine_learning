function [m_database,V_PCA,ProjectedImages_PCA] = FisherfaceCore(T)
%m_database��ѵ������������о�ֵ  V_PCA��PCA�ռ䴫�ݾ���  
%V_Fisher����������J������C-1����������  ProjectedImages_Fisher��fisher�����ӿռ�
        

%ѵ����ͼƬz1
Class_number = ( size(T,2) )/2;                  %��������������
Class_population = 2;                               %ÿһ�ࣨÿ���ˣ���ͼƬ��
Pic = Class_population * Class_number;      %��ͼƬ��

m_database = mean(T,2);                         %ѵ������������о�ֵ

A = T - repmat(m_database,1,Pic);        %�������A��M*N x Pic��������ÿһ��ѵ����ͼƬ�����ƽ��ֵ�ı���̶�
L = A'*A;                 % Э�������Pic x Pic��
[V,D] = eig(L);         % V��������������  D������ֵ�Խ���


%���˵�С������ֵ��Ӧ����������
L_eig_vec = [];
for i = 1 : Pic-Class_number 
    L_eig_vec = [L_eig_vec V(:,i)];
end

%V_PCA��PCA�ռ䴫�ݾ���M*N x Pic-C��  
V_PCA =   A * L_eig_vec; 

%ͶӰ��PCA�����ռ�
ProjectedImages_PCA = [];
for i = 1 : Pic
    temp = V_PCA'*A(:,i);   
    ProjectedImages_PCA = [ProjectedImages_PCA temp];  %PCA�����ӿռ䣨Pic-C x Pic��
end