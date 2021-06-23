function [OutputName,Difference] = Recognition(TestImage, m_database,V_PCA, ProjectedImages_PCA)
          

Train_Number = size(ProjectedImages_PCA,2);
%��ȡ����ͼƬ��FLD����
InputImage = imread(TestImage);
temp = InputImage(:,:,1);

[irow,icol] = size(temp);
InImage = reshape(temp',irow*icol,1);
Difference = double(InImage)-m_database;       %����̶�
ProjectedTestImage = V_PCA' * Difference;      % ����ͼƬ����������

%����ŷʽ���룬ȡŷʽ������С��ͼƬ���
Euc_dist = [];
for i = 1 : Train_Number
    q = ProjectedImages_PCA(:,i);
    temp = ( norm( ProjectedTestImage - q ) )^2;
    Euc_dist = [Euc_dist temp]; 
end

[Euc_dist_min , Recognized_index] = min(Euc_dist);
OutputName = strcat(int2str(Recognized_index),'.pgm');
