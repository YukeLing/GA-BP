% ��ջ�������
clc
clear
% 
%% ����ṹ����
%��ȡ����
input=[106 126 141 126 127 ...
    113 123 109 102 97 89 94 103 97 127 152 200 296 317 301 353 267 217 257 ...
    293 215 192 155 146 150 153 160 158 438 11012 12981 14392 9965 6916 5535 ...
    7630 9613 10241;214 250 272 272 296 280 308 305 328 325 329 344 382 393 428 488 558 ...
    668 736 755 824 767 858 901 760 800 802 777 805 852 841 870 890 1142 ...
    11728 13716 15135 10755 7796 6435 8614 10651 11109]';
output=[108,125,131,146,169,166,185,196,226,229,240,250,279,296,...
    301,335,358,372,419,454,470,500,641,644,467,585,610,622,658,702,...
    678,710,732,704,716,735,743,790,880,900,984,1038,868]';

%�ڵ����
inputnum=2;
hiddennum=2;
outputnum=1;

%ѵ�����ݺ�Ԥ������
input_train=input(1:37,:)';
input_test=input(38:42,:)';
output_train=output(1:37)';
output_test=output(38:42)';

%ѵ����������������ݹ�һ��
[inputn,inputps]=mapminmax(input_train);
[outputn,outputps]=mapminmax(output_train);

%��������
net=newff(inputn,outputn,hiddennum);

%% �Ŵ��㷨������ʼ��
maxgen=50;                         %��������������������
sizepop=10;                        %��Ⱥ��ģ
pcross=[0.4];                       %�������ѡ��0��1֮��
pmutation=[0.2];                    %�������ѡ��0��1֮��

%�ڵ�����
numsum=inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum;

lenchrom=ones(1,numsum);                       %���峤��
bound=[-3*ones(numsum,1) 3*ones(numsum,1)];    %���巶Χ

individuals=struct('fitness',zeros(1,sizepop), 'chrom',[]);  %����Ⱥ��Ϣ����Ϊһ���ṹ��
avgfitness=[];                      %ÿһ����Ⱥ��ƽ����Ӧ��
bestfitness=[];                     %ÿһ����Ⱥ�������Ӧ��
bestchrom=[];                       %��Ӧ����õ�Ⱦɫ��
%���������Ӧ��ֵ
for i=1:sizepop
    %�������һ����Ⱥ
    individuals.chrom(i,:)=Code(lenchrom,bound);    %���루binary��grey�ı�����Ϊһ��ʵ����float�ı�����Ϊһ��ʵ��������
    x=individuals.chrom(i,:);
    %������Ӧ��
    individuals.fitness(i)=fun(x,inputnum,hiddennum,outputnum,net,inputn,outputn);   %Ⱦɫ�����Ӧ��
end
FitRecord=[];
%����õ�Ⱦɫ��
[bestfitness bestindex]=min(individuals.fitness);
bestchrom=individuals.chrom(bestindex,:);  %��õ�Ⱦɫ��
avgfitness=sum(individuals.fitness)/sizepop; %Ⱦɫ���ƽ����Ӧ��
%��¼ÿһ����������õ���Ӧ�Ⱥ�ƽ����Ӧ��
trace=[avgfitness bestfitness]; 

%% ���������ѳ�ʼ��ֵ��Ȩֵ
% ������ʼ
for i=1:maxgen
  
    % ѡ��
    individuals=Select(individuals,sizepop); 
    avgfitness=sum(individuals.fitness)/sizepop;
    %����
    individuals.chrom=Cross(pcross,lenchrom,individuals.chrom,sizepop,bound);
    % ����
    individuals.chrom=Mutation(pmutation,lenchrom,individuals.chrom,sizepop,i,maxgen,bound);
    
    % ������Ӧ�� 
    for j=1:sizepop
        x=individuals.chrom(j,:); %����
        individuals.fitness(j)=fun(x,inputnum,hiddennum,outputnum,net,inputn,outputn);   
    end
    
    %�ҵ���С�������Ӧ�ȵ�Ⱦɫ�弰��������Ⱥ�е�λ��
    [newbestfitness,newbestindex]=min(individuals.fitness);
    [worestfitness,worestindex]=max(individuals.fitness);
    
    %���Ÿ������
    if bestfitness>newbestfitness
        bestfitness=newbestfitness;
        bestchrom=individuals.chrom(newbestindex,:);
    end
    individuals.chrom(worestindex,:)=bestchrom;
    individuals.fitness(worestindex)=bestfitness;
    
    %��¼ÿһ����������õ���Ӧ�Ⱥ�ƽ����Ӧ��
    avgfitness=sum(individuals.fitness)/sizepop;
    trace=[trace;avgfitness bestfitness]; 
    FitRecord=[FitRecord;individuals.fitness];
end

%% �����ų�ʼ��ֵȨֵ��������Ԥ��
% %���Ŵ��㷨�Ż���BP�������ֵԤ��
w1=x(1:inputnum*hiddennum);
B1=x(inputnum*hiddennum+1:inputnum*hiddennum+hiddennum);
w2=x(inputnum*hiddennum+hiddennum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum);
B2=x(inputnum*hiddennum+hiddennum+hiddennum*outputnum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum);

net.iw{1,1}=reshape(w1,hiddennum,inputnum);
net.lw{2,1}=reshape(w2,outputnum,hiddennum);
net.b{1}=reshape(B1,hiddennum,1);
net.b{2}=B2;

%% BP����ѵ��
%�����������
net.trainParam.epochs=100;
net.trainParam.lr=0.01;
%net.trainParam.goal=0.00001;

%����ѵ��
[net,per2]=train(net,inputn,outputn);

%% BP����Ԥ��
%���ݹ�һ��
inputn_test=mapminmax('apply',input_test,inputps);
an=sim(net,inputn_test);
test_simu=mapminmax('reverse',an,outputps);
error=test_simu-output_test;



%% �Ŵ��㷨������� 
figure(1)

[r c]=size(trace);
plot([1:r]',trace(:,2),'b--');
title(['��Ӧ������  ' '��ֹ������' num2str(maxgen)]);
xlabel('��������');ylabel('��Ӧ��');
legend('ƽ����Ӧ��','�����Ӧ��');
% disp('��Ӧ��                   ����');

%% GA�Ż�BP����Ԥ��������
figure(2)

plot(test_simu,':og')
hold on
plot(output_test,'-*');
legend('Ԥ�����','�������')
title('GA�Ż�BP����Ԥ�����','fontsize',12)
ylabel('�������','fontsize',12)
xlabel('����','fontsize',12)
%Ԥ�����
error=test_simu-output_test;

figure(3)

plot(error,'-*')
title('GA�Ż�BP������Ԥ�����','fontsize',12)
ylabel('���','fontsize',12)
xlabel('����','fontsize',12)

figure(4)

plot((test_simu-output_test)./output_test,'-*');
title('GA�Ż�BP������Ԥ�����ٷֱ�')

errorsum=sum(abs(error));
disp(errorsum/(length(output_test)));