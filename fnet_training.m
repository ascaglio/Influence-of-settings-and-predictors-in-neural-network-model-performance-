pm=medfilt1(pm25acu); %Median filter to target 

input= [PM10acu NOx BLH SH2 u v]; %Inputs to neural network 

%PM10acu: PM10 from ACUMAR
%NOx: nitrogen oxides
%BLH: boundary layer height
%SH2: hydrogen sulfide
%u: wind zonal component 
%v: wind meridional component
           
in=transpose(input); %time advance in columns

%Creation of network and architecture
net=network;
net.name='FNET';
net.numInputs=6; %number of predictors
net.numLayers=2; %number of layers 
net.biasConnect=[1;0]; %bias connections 
net.inputConnect=[1 1 1 1 1 1;0 0 0 0 0 0]; %inputs connections
net.layerConnect=[0 1;1 0]; %Connections between layers
net.outputConnect=[0 1]; %Conextion between hidden and output layers
net.layerWeights{1,2}.delays =1; %delay to feedback

%Input layer
net.inputs{1}.exampleInput=in(1,:);      %input data
net.inputs{1}.name='PM10acu';            %predictor name                                                      
net.inputs{2}.exampleInput=in(2,:);
net.inputs{2}.name='NOx';
net.inputs{3}.exampleInput=in(3,:);
net.inputs{3}.name='BLH';
net.inputs{4}.exampleInput=in(4,:);
net.inputs{4}.name='SH2';
net.inputs{5}.exampleInput=in(5,:);
net.inputs{5}.name='U';
net.inputs{6}.exampleInput=in(6,:);
net.inputs{6}.name='V';

for i=1:1:6
   net.inputs{i}.processFcns={'mapstd'}; % pre-processing
end

net.initFcn='initlay'; %initialization of layers

%Hidden layer
net.layers{1}.name='hidden layer';
net.layers{1}.size=20; %Number of neurons
net.layers{1}.transferFcn='radbas'; %Transfer function 
net.layers{1}.initFcn='initnw'; %Initialization function for bias and weights

%Output layer
net.layers{2}.name='output layer';
net.layers{2}.size=1;                 %Number of output neurons 
net.layers{2}.transferFcn='purelin';  %Transfer function
net.layers{2}.initFcn='initnw';  %Initialization function

net.performFcn='rmse'; %RMSE
net.trainFcn='traincgf'; %Training function
net.divideFcn='dividerand'; %Percentage of data for training, validation and test
net.plotFcns={'plotperform','plottrainstate', 'plotregression'}; %Graphics

net=init(net); %Initialization of network

target=pm; %Selection of target
tar=transpose(target); %Arrangement of data according to input data configuration
[T,PS2]=mapstd(tar); % Pre-processing of target 


%Training hyper-parameters and early stopping settings

net.trainParam.epochs=1000; %Epochs
net.trainParam.goal=0;       %Performance goal
net.trainParam.max_fail=6;   %Máximum number of fails 
net.trainParam.min_grad=1e-7; %Minimum gradient of performance
net.trainParam.mu=0.001; %Gain in initial training
net.trainParam.mu_dec=0.1; %Decreasing gain factor
net.trainParam.mu_inc=10;    %Increasing gain factor
net.trainParam.mu_max=1e10;     %Maximum gain
net.trainParam.time=inf;     %Maximum time for training 
net.trainParam.showCommandLine = true;   %Show command lines 
net.trainParam.show= 2;   %Show indicators every 2 iterations

[net,tr]=train(net,in,T); %Start training



