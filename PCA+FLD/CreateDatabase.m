function T = CreateDatabase(TrainDatabasePath)
         

TrainFiles = dir(TrainDatabasePath);
Train_Number = 0;

for i = 1:size(TrainFiles,1)
    if not(strcmp(TrainFiles(i).name,'.')|strcmp(TrainFiles(i).name,'..')|strcmp(TrainFiles(i).name,'Thumbs.db'))
        Train_Number = Train_Number + 1; 
    end
end


T = [];
for i = 1 : Train_Number
    str = int2str(i);
    str = strcat('\',str,'.pgm');
    str = strcat(TrainDatabasePath,str);
    
    img = imread(str);
    
    [irow,icol] = size(img);
   
    temp = reshape(img',irow*icol,1);           % 将每一张二维训练图片转换为一维
    T = [T temp];              
end
T = double(T);