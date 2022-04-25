matriz=input_som;  %Input space
sD=som_data_struct(matriz,'comp_names',{'PM10apra','SO2','SH2','NOx','O3','MPxi','Oxi','PM10acu','PM25acu','HCNM','HCM','RH','U','P','PPT','rad','DP','stab','hour','weekday','month'});  %Data structure from input space
sDnor=som_normalize(sD,'var'); %Normalization of input space

sM1=som_make(sDnor,'msize',[15 15],'name','SOM'); %Initialization and training of network

figure
som_show(sM1,'umati','','bar','none','comp',1:25'norm','n'); %figure of wights for input variables in final map
colorbar

[qe,te] = som_quality(sM1,sDnor); %quantization and topographic error
[adm,admu,tdmu] = som_distortion(sM1,sDnor); %distortion of final map
