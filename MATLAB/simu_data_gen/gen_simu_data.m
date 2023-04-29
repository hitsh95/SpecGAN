clear all;
warning off

root_path = './';
filename = 'solvent.xls';

raw_data = readxlsfile(filename);

%% read tiff file

NumData = size(raw_data.cls,2);
lambda = linspace(560,679,120);

gt_data.cls = raw_data.cls;
gt_data.clsName=raw_data.clsName;
gt_data.lambda=lambda;
gt_data.spec_data=zeros(length(lambda),NumData);
gt_data.centroid = zeros(1,NumData);

for i = 1:NumData
    spec = raw_data.val(1:end,i);
    
    %双高斯拟合
    modelFun=@(p,x)p(1)*exp(-p(2)*(x-p(3)).^2)+p(4)*exp(-p(5)*(x-p(6)).^2);
    startingVals = [1.0, 0.001, 620, 0.5, 0.001, 590];  

    try
        nlModel = fitnlm(raw_data.lambda,spec,modelFun,startingVals);
        fitRMSE = nlModel.RMSE;
    catch
        disp(raw_data.clsName(i)+" cannot fit, please check it!")
        continue
    end

    fitValue = predict(nlModel,lambda');
    [pks,locs] = max(spec);
    gt_data.spec_data(1:end,i) = fitValue;
    gt_data.centroid(i) = raw_data.lambda(locs);

%     figure(1)
%     plot(locs,pks, '+');
%     line(raw_data.lambda, spec);
%     xlabel('lambda');
%     ylabel('Int');
%     line(lambda', predict(nlModel,lambda'), 'Color', 'r');
end

%       1    2      3       4       5     6     7       8       9       10
% cls=["d","ds31","ds21","dsc311","ds","ds12","ds13","dsc211","dsc","dsc112","dsc113","dsc121","dsc131"];
% plot(gt_data.cls, gt_data.centroid, 'o');
% 588 628 626 621 604

%% generate simulated data

Numpercls = 1000;
simu_data = struct( 'cls', zeros(Numpercls*NumData,1),...
    'id', zeros(Numpercls*NumData,1),...
    'raw_centroid', zeros(Numpercls*NumData,1), ...
    'gt_centroid', zeros(Numpercls*NumData,1), ...
    'clsName', string(zeros(Numpercls*NumData,1)), ...
    'spec_value', zeros(Numpercls*NumData,120), ...
    'res', zeros(Numpercls*NumData,120), ...
    'sigma', zeros(Numpercls*NumData,1), ...
    'ku', zeros(Numpercls*NumData,1), ...
    'cv',zeros(Numpercls*NumData, 1), ...
    'uncertainty',zeros(Numpercls*NumData,1), ...
    'noise_mu',zeros(Numpercls*NumData,1), ...
    'noise_sigma',zeros(Numpercls*NumData,1), ...
    'noiseType',string(zeros(Numpercls*NumData,1)));

for i = 1:NumData
    gt_spec = gt_data.spec_data(1:end,i);
    sigArray = abs(0.1*randn(1,Numpercls));
    muArray = 0.01*randn(1,Numpercls);
    for j = 1:Numpercls
        id = (i-1)*Numpercls+j;
        if ~(mod(id, 500))
            disp("now is the " + string(id) + "th curve")

        end
        norm_noise = imnoise(gt_spec,'gaussian', muArray(j), sigArray(j));
        speckle_noise = imnoise(gt_spec,'speckle', sigArray(j));
        poisson_noise = imnoise(gt_spec, 'poisson');

        tmp = rand();
        if tmp < 0.5 % norm_noise
            spec_data = normalize(norm_noise, 'range');
            simu_data.noise_mu(id) = muArray(j);
            simu_data.noise_sigma(id) = sigArray(j);
            simu_data.noiseType(id) = 'norm';

        elseif tmp > 0.5 && tmp <0.8 % speckle_noise
            spec_data = normalize(speckle_noise,'range');
            simu_data.noise_sigma(id) = sigArray(j);
            simu_data.noiseType(id) = 'speckle';

        else % fuse all noise
            spec_data = normalize(norm_noise + speckle_noise + poisson_noise, 'range');
            simu_data.noise_mu(id) = muArray(j);
            simu_data.noise_sigma(id) = sigArray(j);
            simu_data.noiseType(id) = 'norm+speckle+poisson';
        end

        simu_data.spec_value(id,1:end) = spec_data;

        % VMD分解参数设置
        alpha=2000;  % alpha   - 惩罚因子
        tol=1e-7;    % tol     - 收敛容差，是优化的停止准则之一，可以取 1e-6~5e-6
        K=4;         % K       - 指定分解模态数
        type = 2;    % 采用第三方vmd函数进行分解
        
        % 以下输入参数在使用MATLAB内置函数的时候不需要输入（可以置为nan）
        tau=0;      % tau     - time-step of the dual ascent ( pick 0 for noise-slack )
        DC=1;       % DC      - true if the first mode is put and kept at DC (0-freq)
        init=1;     % init    - 0 = all omegas start at 0
        imf = pVMD(spec_data,length(lambda), alpha, K, tol, type, tau, DC, init); %p文件可调用，无法查看源码，需要源码请获取完整版代码
        close all

        simu_data.res(id,1:end) = imf(4,:);  %residual

        modelFun=@(p,x)p(1)*exp(-p(2)*(x-p(3)).^2)+p(4)*exp(-p(5)*(x-p(6)).^2);
        startingVals = [max(spec_data), 0.001, 620, mean(spec_data), 0.001, 590];  %双高斯拟合
        
        fitRMSE = 0;
        try
            nlModel = fitnlm(lambda,spec_data,modelFun,startingVals);
            fitRMSE = nlModel.RMSE;
        catch
            simu_data.id(id,1:end) = 0;
            continue
        end
    
        fitValue = predict(nlModel,lambda');
        % precise emission maximum was located by fitting a second-order polynomial around 
        % the pixel with maximum intensity. 
        [pks,locs] = findpeaks(fitValue, lambda);
        [photons,maxPos] = max(pks);
        raw_Centroid = locs(maxPos);
    
        % uncertainty
        uncertainty = (max(fitValue) - min(fitValue)) / mean(fitValue);
        % kurtosis
        fitku = kurtosis(fitValue);
        % sigma
        fitSigma = std(fitValue);
        % Coefficient of Variation
        fitCv = fitSigma / mean(fitValue);
    
    %     figure(5)
    %     plot(lambda, spec_data, 'o', locs,pks, '+');
    %     line(lambda, spec_data);
    %     xlabel('lambda');
    %     ylabel('Int');
    %     line(lambda, predict(nlModel,lambda), 'Color', 'r');

        if fitCv<0.4 || isempty(pks) || fitRMSE>4e3%% 
            continue
        end
    
        simu_data.id(id) = id;
        simu_data.cls(id) = raw_data.cls(i);
        simu_data.clsName(id) = raw_data.clsName(i);
        simu_data.gt_centroid(id) = gt_data.centroid(i);
        simu_data.uncertainty(id) = uncertainty;
        simu_data.ku(id) = fitku;
        simu_data.sigma(id) = fitSigma;
        simu_data.cv(id) = fitCv;
        simu_data.raw_centroid(id) = raw_Centroid;
        
    end
end
%%
ind = find(simu_data.id ~= 0);

simu_data.id = simu_data.id(ind);
simu_data.cls = simu_data.cls(ind);
simu_data.clsName = simu_data.clsName(ind);
simu_data.uncertainty = simu_data.uncertainty(ind);
simu_data.ku = simu_data.ku(ind);
simu_data.cv = simu_data.cv(ind);
simu_data.sigma = simu_data.sigma(ind);
simu_data.gt_centroid = simu_data.gt_centroid(ind);
simu_data.raw_centroid = simu_data.raw_centroid(ind);
simu_data.noise_mu = simu_data.noise_mu(ind);
simu_data.noise_sigma = simu_data.noise_sigma(ind);
simu_data.noiseType = simu_data.noiseType(ind);
simu_data.spec_value = simu_data.spec_value(ind,:);
simu_data.res = simu_data.res(ind,:);

% disrupt the order
randIndex = randperm(size(simu_data.id,1));

simu_data.id = simu_data.id(randIndex);
simu_data.cls = simu_data.cls(randIndex);
simu_data.clsName = simu_data.clsName(randIndex);
simu_data.uncertainty = simu_data.uncertainty(randIndex);
simu_data.ku = simu_data.ku(randIndex);
simu_data.cv = simu_data.cv(randIndex);
simu_data.sigma = simu_data.sigma(randIndex);
simu_data.gt_centroid = simu_data.gt_centroid(randIndex);
simu_data.raw_centroid = simu_data.raw_centroid(randIndex);
simu_data.noise_mu = simu_data.noise_mu(randIndex);
simu_data.noise_sigma = simu_data.noise_sigma(randIndex);
simu_data.noiseType = simu_data.noiseType(randIndex);
simu_data.spec_value = simu_data.spec_value(randIndex,:);
simu_data.res = simu_data.res(randIndex,:);

%% plot averaged spectrum 
figure(2)
set(gcf, 'Position', [100 100 900 600]); 
t = tiledlayout(2,3,'TileSpacing','Compact');

ind = find(simu_data.cls==7);

% average specrum
nexttile
y = mean(simu_data.spec_value(ind,:),1);
plot(lambda, y, 'r-','LineWidth',1.2);
title('Averaged Spectrum', 'FontSize', 14, 'FontWeight','bold');
xlabel('Wavelength [nm]', 'FontSize', 14, 'FontWeight','bold') ;
ylabel('Fluoresence Intensity', 'FontSize', 14, 'FontWeight','bold') ;

% Centroid Histogram
nexttile
histogram(simu_data.raw_centroid(ind));
title('Centroid Histogram', 'FontSize', 14, 'FontWeight','bold');
xlabel('Wavelength [nm]', 'FontSize', 14, 'FontWeight','bold') ;
ylabel('Count', 'FontSize', 14, 'FontWeight','bold') ;

% Centroid Vs Sigma
nexttile
scatter(simu_data.raw_centroid(ind), simu_data.sigma(ind), '.');
title('Centroid Vs Sigma', 'FontSize', 14, 'FontWeight','bold');
xlabel('Centroid', 'FontSize', 14, 'FontWeight','bold') ;
ylabel('Sigma', 'FontSize', 14, 'FontWeight','bold') ;

% Centroid Vs kurtosis
nexttile
scatter(simu_data.raw_centroid(ind), simu_data.ku(ind), '.');
title('Centroid Vs kurtosis', 'FontSize', 14, 'FontWeight','bold');
xlabel('Centroid', 'FontSize', 14, 'FontWeight','bold') ;
ylabel('kurtosis', 'FontSize', 14, 'FontWeight','bold') ;

% Centroid Vs Variation Coefficient
nexttile
scatter(simu_data.raw_centroid(ind), simu_data.cv(ind), '.');
title('Centroid Vs CV', 'FontSize', 14, 'FontWeight','bold');
xlabel('Centroid', 'FontSize', 14, 'FontWeight','bold') ;
ylabel('Coefficient of Variation', 'FontSize', 14, 'FontWeight','bold') ;

nexttile
plot(lambda, gt_data.spec_data(:,4), 'r-','LineWidth',1.2);
title('ground truth', 'FontSize', 14, 'FontWeight','bold');
xlabel('lambda(nm)', 'FontSize', 14, 'FontWeight','bold') ;
ylabel('norm Int', 'FontSize', 14, 'FontWeight','bold') ;

%% save to csv file

% traning datset
Index = 1:floor(size(simu_data.id,1) * 0.8);

savecsv(root_path, "simu_spec_train.csv", simu_data, Index, lambda, "raw");
savecsv(root_path, "simu_spec_train_res.csv", simu_data, Index, lambda, "res");

% validated dataset
train_len = floor(size(simu_data.id,1) * 0.8);
Index = train_len +1: train_len + floor(size(simu_data.id,1) * 0.1);

savecsv(root_path, "simu_spec_val.csv", simu_data, Index, lambda, "raw");
savecsv(root_path, "simu_spec_val_res.csv", simu_data, Index, lambda, "res");

% test dataset
train_val_len = floor(size(simu_data.id,1) * 0.9);
Index = train_val_len +1: train_val_len + floor(size(simu_data.id,1) * 0.1);

savecsv(root_path, "simu_spec_test.csv", simu_data, Index, lambda, "raw");
savecsv(root_path, "simu_spec_test_res.csv", simu_data, Index, lambda, "res");

% ground truth data
gt_filename = strcat(root_path, 'gt.csv');
csvtitle = {'cls', 'clsName', 'centroid'};
spec_lambda = string(lambda);

spec_table = table(gt_data.cls', gt_data.clsName', gt_data.centroid', 'VariableNames', csvtitle);
for i=1:length(lambda)
    tab = table(gt_data.spec_data(i,:)', 'VariableNames', spec_lambda(i));
    spec_table = [spec_table tab];
end

writetable(spec_table, gt_filename);



%%
figure(1)
set(gcf, 'Position', [100 100 900 600]); 
% ind = find(simu_data.cls==1);
% average specrum
lambda = linspace(560,659,991);
for i=1:5
    plot(lambda,raw_data.val(1:991,i),'LineWidth',1.2)
    title('Averaged Spectrum', 'FontSize', 14, 'FontWeight','bold');
    xlabel('Wavelength [nm]', 'FontSize', 14, 'FontWeight','bold') ;
    ylabel('Fluoresence Intensity', 'FontSize', 14, 'FontWeight','bold') ;
    hold on
end
    legend(raw_data.clsName)

figure(2)
set(gcf, 'Position', [100 100 900 600]); 
% ind = find(simu_data.cls==1);
% average specrum
lambda = linspace(560,659,100);
for i=1:5
    plot(lambda,gt_data.spec_data(1:100,i),'LineWidth',1.2)
    title('Averaged Spectrum', 'FontSize', 14, 'FontWeight','bold');
    xlabel('Wavelength [nm]', 'FontSize', 14, 'FontWeight','bold') ;
    ylabel('Fluoresence Intensity', 'FontSize', 14, 'FontWeight','bold') ;
    hold on
end
    legend(raw_data.clsName)

% 
figure(3)

X = categorical({'acetone','CHCL3','C3H8O','MeOH','EtOH'});
X = reordercats(X,{'acetone','CHCL3','C3H8O','MeOH','EtOH'});
Y = gt_data.centroid';
b = bar(X, Y);
ylim([580 680])

xtips1 = b(1).XEndPoints;
ytips1 = b(1).YEndPoints;
labels1 = string(b(1).YData);
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')

title('Fit Centroid', 'FontSize', 14, 'FontWeight','bold');
xlabel('Sovlent Composition', 'FontSize', 14, 'FontWeight','bold') ;
ylabel('Lambda [nm]', 'FontSize', 14, 'FontWeight','bold') ;