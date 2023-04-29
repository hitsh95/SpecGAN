clear all;
warning off
root_path = '../../data/fc_data/';
filename = 'fc';

specFile = strcat(root_path, filename, "_spec_test.csv");
spec_data = ReadMatlabCsvFile(specFile);
%%
NumData = length(spec_data.id);
NumLamb = length(spec_data.lambda);
lambda = spec_data.lambda;
spec_data.res=zeros(NumData,NumLamb);
spec_data.fit_spec_value=zeros(NumData,NumLamb);

for i=1:NumData
    
    norm_spec_value = spec_data.norm_spec_value(i,:);

    % VMD分解参数设置
    alpha=2000;  % alpha   - 惩罚因子
    tol=1e-7;    % tol     - 收敛容差，是优化的停止准则之一，可以取 1e-6~5e-6
    K=4;         % K       - 指定分解模态数
    type = 2;    % 采用第三方vmd函数进行分解
    
    % 以下输入参数在使用MATLAB内置函数的时候不需要输入（可以置为nan）
    tau=0;      % tau     - time-step of the dual ascent ( pick 0 for noise-slack )
    DC=1;       % DC      - true if the first mode is put and kept at DC (0-freq)
    init=1;     % init    - 0 = all omegas start at 0
    imf = pVMD(norm_spec_value,length(spec_data.lambda), alpha, K, tol, type, tau, DC, init); %p文件可调用，无法查看源码，需要源码请获取完整版代码
    close all

    spec_data.res(i,1:end) = normalize(imf(4,:),'range');  %residual

    if ~mod(i,500)
        disp("now is process the " + string(i) +"/" + string(NumData)+ " epoch...");
    end

end
%%
% disrupt the order
randIndex = randperm(size(spec_data.id,1));

spec_data.id = spec_data.id(randIndex);
spec_data.cls = spec_data.cls(randIndex);
spec_data.clsName = spec_data.clsName(randIndex);
spec_data.ku = spec_data.ku(randIndex);
spec_data.cv = spec_data.cv(randIndex);
spec_data.s_px = spec_data.s_px(randIndex);
spec_data.s_py = spec_data.s_py(randIndex);
spec_data.gt_centroid = spec_data.gt_centroid(randIndex);
spec_data.raw_centroid = spec_data.raw_centroid(randIndex);
spec_data.spec_value = spec_data.spec_value(randIndex,:);
spec_data.res = spec_data.res(randIndex,:);
spec_data.norm_spec_value = spec_data.norm_spec_value(randIndex,:);

%% plot averaged spectrum 
figure(1)
set(gcf, 'Position', [100 100 1200 600]); 
t = tiledlayout(1,2,'TileSpacing','Compact');

Numcls = 5;
res_data.spec_value = zeros(Numcls,120);
res_data.gt_centroid = zeros(Numcls,1);
res_data.cls = Inf(Numcls,1);
res_data.clsName = string(zeros(Numcls,1));

cls = [0,2,5,7,8];

for i = 1:Numcls
    
    nexttile(1)
    ind = find(spec_data.cls==cls(i));
    y = normalize(mean(spec_data.spec_value(ind,:),1),'range');
    legend_name = spec_data.clsName(ind(1));
    plot(lambda,y,'LineWidth',1.2, 'DisplayName', legend_name)
    title('Averaged Spectrum', 'FontSize', 14, 'FontWeight','bold');
    xlabel('Wavelength [nm]', 'FontSize', 14, 'FontWeight','bold') ;
    ylabel('Fluoresence Intensity', 'FontSize', 14, 'FontWeight','bold') ;
    hold on
    legend

    nexttile(2)
    y = normalize(mean(spec_data.res(ind,:),1), 'range');
    plot(lambda,y,'LineWidth',1.2, 'DisplayName', legend_name)
    title('Averaged Spectrum (res)', 'FontSize', 14, 'FontWeight','bold');
    xlabel('Wavelength [nm]', 'FontSize', 14, 'FontWeight','bold') ;
    ylabel('Fluoresence Intensity', 'FontSize', 14, 'FontWeight','bold') ;
    hold on
    legend

    modelFun=@(p,x)p(1)*exp(-p(2)*(x-p(3)).^2)+p(4)*exp(-p(5)*(x-p(6)).^2);
    startingVals = [max(y), 0.001, 590, mean(y), 0.001, 650];  %双高斯拟合
    try
        nlModel = fitnlm(lambda,y,modelFun,startingVals);
        fitRMSE = nlModel.RMSE;
    catch
        continue
    end
    fitValue = predict(nlModel,spec_data.lambda');
    [pks,locs] = findpeaks(fitValue, spec_data.lambda);
    [photons,maxPos] = max(pks);
    res_centroid = locs(maxPos);

    res_data.spec_value(i,:) = fitValue;
    res_data.gt_centroid(i,1) = res_centroid;
    res_data.cls(i,1) = i;
    res_data.clsName(i,1) = legend_name;
end


figure(2)
gt_filename = strcat(root_path, 'gt.csv');
gt_data = ReadMatlabCsvFile(gt_filename);

X = categorical({'sc','dsc311','dsc','ds','d'});
X = reordercats(X,{'sc','dsc311','dsc','ds','d'});
res_Y = res_data.gt_centroid';
raw_Y = gt_data.gt_centroid';
Y = [raw_Y; res_Y];
b = bar(X, Y);
ylim([580 680])
legend('raw','res')

xtips1 = b(1).XEndPoints;
ytips1 = b(1).YEndPoints;
labels1 = string(b(1).YData);
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')
xtips2 = b(2).XEndPoints;
ytips2 = b(2).YEndPoints;
labels2 = string(b(2).YData);
text(xtips2,ytips2,labels2,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')

title('Fit Centroid', 'FontSize', 14, 'FontWeight','bold');
xlabel('Lipid Composition', 'FontSize', 14, 'FontWeight','bold') ;
ylabel('Lambda [nm]', 'FontSize', 14, 'FontWeight','bold') ;



%% save to csv file

% traning datset
Index = 1:floor(size(spec_data.id,1) * 0.8);

savecsv(root_path, filename + "_spec_train.csv", spec_data, Index, lambda, "raw");
savecsv(root_path, filename + "_spec_train_res.csv", spec_data, Index, lambda, "res");

% validated dataset
train_len = floor(size(spec_data.id,1) * 0.8);
Index = train_len +1: train_len + floor(size(spec_data.id,1) * 0.1);

savecsv(root_path, filename + "_spec_val.csv", spec_data, Index, lambda, "raw");
savecsv(root_path, filename + "_spec_val_res.csv", spec_data, Index, lambda, "res");

% test dataset
train_val_len = floor(size(spec_data.id,1) * 0.9);
Index = train_val_len +1: train_val_len + floor(size(spec_data.id,1) * 0.1);

savecsv(root_path, filename + "_spec_test.csv", spec_data, Index, lambda, "raw");
savecsv(root_path, filename + "_spec_test_res.csv", spec_data, Index, lambda, "res");




