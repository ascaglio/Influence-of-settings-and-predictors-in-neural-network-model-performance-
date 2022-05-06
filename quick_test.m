input= input(:,1); %Input to neural network. PM10acu: PM10 from ACUMAR
 
in=transpose(input); %time advance in columns

%Creation of network and architecture
net=network;
net.name='TEST';
net.numInputs=1; %number of predictors
net.numLayers=2; %number of layers 
net.biasConnect=[1;0]; %bias connections 
net.inputConnect=[1;0]; %inputs connections
net.layerConnect=[0 0;1 0]; %Connections between layers
net.outputConnect=[0 1]; %Conextion between hidden and output layers

%Input layer
net.inputs{1}.exampleInput=in(1,:);      %input data
net.inputs{1}.name='PM10acu';            %predictor name                                                      

net.inputs{1}.processFcns={'mapstd'}; % pre-processing

net.initFcn='initlay'; %initialization of layers

%Hidden layer
net.layers{1}.name='hidden layer';
net.layers{1}.size=10; %Number of neurons
net.layers{1}.transferFcn='tansig'; %Transfer function 
net.layers{1}.initFcn='initnw'; %Initialization function for bias and weights

%Output layer
net.layers{2}.name='output layer';
net.layers{2}.size=1;                 %Number of output neurons 
net.layers{2}.transferFcn='purelin';  %Transfer function
net.layers{2}.initFcn='initnw';  %Initialization function

net.performFcn='mse'; %MSE
net.trainFcn='trainlm'; %Training function
net.divideFcn='dividerand'; %Percentage of data for training, validation and test
net.plotFcns={'plotperform','plottrainstate', 'plotregression'}; %Graphics

net=init(net); %Initialization of network

target=pm25acu; %Selection of target
tar=transpose(target); %Arrangement of data according to input data configuration
[T,PS2]=mapstd(tar); % Pre-processing of target 

[net,tr]=train(net,in,T); %Start training


