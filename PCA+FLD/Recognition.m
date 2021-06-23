function [OutputName,Difference] = Recognition(TestImage, m_database,V_PCA, ProjectedImages_PCA)
          

Train_Number = size(ProjectedImages_PCA,2);
%提取测试图片的FLD特征
InputImage = imread(TestImage);
temp = InputImage(:,:,1);

[irow,icol] = size(temp);
InImage = reshape(temp',irow*icol,1);
Difference = double(InImage)-m_database;       %背离程度
ProjectedTestImage = V_PCA' * Difference;      % 测试图片的特征向量

%计算欧式距离，取欧式距离最小的图片输出
Euc_dist = [];
for i = 1 : Train_Number
    q = ProjectedImages_PCA(:,i);
    temp = ( norm( ProjectedTestImage - q ) )^2;
    Euc_dist = [Euc_dist temp]; 
end

[Euc_dist_min , Recognized_index] = min(Euc_dist);
OutputName = strcat(int2str(Recognized_index),'.pgm');
