%% 此程序为matlab编程实现的BP神经网络
% 清空环境变量
clear
close all
clc
% 自己数据
input=[214 250 272 272 296 280 308 305 328 325 329 344 382 393 428 488 558 668 736 755 824 767 858 901 760 800 802 ...
    777 805 852 841 870 890 1142 11728 13716 15135 10755 7796 6435 8614 10651 11109;106 126 141 126 127 113 123 109 ...
    102 97 89 94 103 97 127 152 200 296 317 301 353 267 217 257 293 215 192 155 146 150 153 160 158 438 11012 12981 14392 9965 6916 5535 7630 9613 10241]';
output=[108 125 131 146 169 166 185 196 226 229 240 250 279 296 301 335 358 372 419 454 470 500 641 644 467 585 610 622 658 702 678 710 732 704 716 735 743 790 880 900 984 1038 868]';

% input=[
%     1104.00 1137.00 1152.00 1168.00 1186.00 1201.00 1217.00 1233.00 1249.00 1265.00 1288.00 1311.00 1334.00 1350.00 1365.00 1381.00 1398.00 1414.00 1451.00 1489.00 1527.00 1567.00 1608.60 1668.33 1712.97 1765.84 1834.98 1890.26 1964.11 2063.58 2140.65 2210.28 2302.66 2355.53 2398.50 2448.43 2467.06 2457.59 2467.37 2466.28 2475.39 2481.34 2488.36;...
%     2333 3151 3402 2546 2821 2733 3890 5700 5983 6341 5117 5276 4921 4980 4961 5756 6993 9019 9522 12735 15968 17326 23189 24672 29517 31272 34571 39539 44888 47739 56485 67465 74658 78213 82454 98266 98857 94310 81246 86246 80360 62554 55309;...
%     54.1 68.28 80.43 88.73 89.80 100.68 123.72 173.39 196.84 225.25 295.83 331.38 333.86 382.06 464.82 679.91 844.64 1069.67 1287.96 1478.23 1650.52 1794.74 1955.17 2126.01 2337.44 2565.20 2863.17 3230.66 3681.66 4250.23 5053.35 5786.83 6901.39 8052.21 8833.20 9693.15 10592.68 11605.70 12588.21 13699.52 14874.76 15847.55 15932.50;...
%      272.81 286.43 311.89 324.76 337.07 351.81 390.85 466.75 490.83 545.46 648.30 696.54 781.66 893.77 1114.32 1519.23 1990.86 2518.08 2980.75 3465.28 3831.00 4222.30 4812.15 5257.66 5795.02 6804.04 8101.55 9197.13 10598.86 12878.68 14536.90 15742.44 17915.41 20009.68 21305.59 23204.12 25269.75 26887.02 29887.02 32925.01 36011.82 37987.55 38700.58;...
%    
%     106 126 141 126 127 113 123 109 102 97 89 94 103 97 127 152 200 296 317 301 353 267 217 257 293 215 192 155 146 150 153 160 158 438 11012 12981 14392 9965 6916 5535 7630 9613 10241;...
%     214 250 272 272 296 280 308 305 328 325 329 344 382 393 428 488 558 668 736 755 824 767 858 901 760 800 802 777 805 852 841 870 890 1142 11728 13716 15135 10755 7796 6435 8614 10651 11109]';
% output=[108 125 131 146 169 166 185 196 226 229 240 250 279 296 301 335 358 372 419 454 470 500 641 644 467 585 ...
%     610 622 658 702 678 710 732 704 716 735 743 790 880 900 984 1038 868]';
%% 第二步 设置训练数据和预测数据
input_train = input(1:38,:)';
output_train =output(1:38,:)';
input_test = input(39:43,:)';
output_test =output(39:43,:)';
%节点个数
inputnum=2; % 输入层节点数量
hiddennum=3;% 隐含层节点数量
outputnum=1; % 输出层节点数量
%% 第三本 训练样本数据归一化
[inputn,inputps]=mapminmax(input_train);%归一化到[-1,1]之间，inputps用来作下一次同样的归一化
[outputn,outputps]=mapminmax(output_train);
%% 第四步 构建BP神经网络
net=newff(inputn,outputn,hiddennum,{'tansig','purelin'},'trainlm');% 建立模型，传递函数使用purelin，采用梯度下降法训练

W1= net.iw{1, 1};%输入层到中间层的权值
B1 = net.b{1};%中间各层神经元阈值

W2 = net.lw{2,1};%中间层到输出层的权值
B2 = net.b{2};%输出层各神经元阈值

%% 第五步 网络参数配置（ 训练次数，学习速率，训练目标最小误差等）
net.trainParam.epochs=1000;         % 训练次数，这里设置为1000次
net.trainParam.lr=0.01;                   % 学习速率，这里设置为0.01
net.trainParam.goal=0.00001;                    % 训练目标最小误差，这里设置为0.00001

%% 第六步 BP神经网络训练
net=train(net,inputn,outputn);%开始训练，其中inputn,outputn分别为输入输出样本

%% 第七步 测试样本归一化
inputn_test=mapminmax('apply',input_test,inputps);% 对样本数据进行归一化

%% 第八步 BP神经网络预测
an=sim(net,inputn_test); %用训练好的模型进行仿真

%% 第九步 预测结果反归一化与误差计算     
test_simu=mapminmax('reverse',an,outputps); %把仿真得到的数据还原为原始的数量级
error=test_simu-output_test;      %预测值和真实值的误差
year=[2016:2020];
%% 第十步 真实值与预测值误差比较
figure('units','normalized','position',[0.119 0.2 0.38 0.5])

plot(year,output_test,'bo-')
hold on
plot(year,test_simu,'r*-')
hold on
plot(year,error,'square','MarkerFaceColor','b')
legend('期望值','预测值','误差')
xlabel('预测年份')
ylabel('垃圾制造量（单位：万吨）')
title('BP神经网络测试集的预测值与实际值对比图')
% 2016,2017,2018,2019,2020
% figure(2)
% 
% plot(error,'-*')
% title('GA优化BP神经网络预测误差','fontsize',12)
% ylabel('误差','fontsize',12)
% xlabel('样本','fontsize',12)
aberror=(test_simu-output_test)./output_test;
figure(2)
plot(year,aberror,'-*');
title('BP神经网络预测误差百分比')
sum_aberror=sum(abs(aberror))/(length(output_test));
[c,l]=size(output_test);
MAE1=sum(abs(error))/(length(output_test));
MSE1=error*error'/1;
RMSE1=MSE1^(1/2);
disp(['-----------------------误差计算--------------------------'])
% disp(['隐含层节点数为',num2str(hiddennum),'时的误差结果如下：'])
ccc=abs(error);
for i=1:length(ccc)
    fprintf('%.2f ',ccc(i));
end
disp(' ')
disp(['平均绝对误差MAE为：',num2str(MAE1)])
disp(['平均百分误差abrrr为：',num2str(sum_aberror)])

