%% load data
clear all;
warning off
root_path = '../../data/lipid_data_new/resnet/';

gan_filename = 'sgan_lipid_outputs';
gt_fileName = 'gt';
raw_fileName = "lipid_spec_test";

gtFile = strcat(root_path, gt_fileName, ".csv");
gt_data = ReadMatlabCsvFile(gtFile);

rawFile = strcat(root_path, raw_fileName, ".csv");
raw_data = ReadMatlabCsvFile(rawFile);

gan_filename = strcat(root_path, gan_filename, ".csv");
gan_data = ReadMatlabCsvFile(gan_filename);

NumData = length(gan_data.id);

for i=1:NumData

    clsName = gan_data.clsName(i);
    ind = find(gan_data.clsName == clsName);
    j = find(gan_data.id(ind)==gan_data.id(i));

    if j > length(ind) - 10
        continue
    end

    spec_value = gan_data.spec_value(ind,:);
    spec = normalize(mean(spec_value(j:j+10,:),1),'range');
    lambda = gan_data.lambda;
    [photons,maxPos] = max(spec);
    fit_centroid = lambda(maxPos);

    % gan_data.fit_spec_value(i,1:end) = fitValue;
    gan_data.fit_centroid(i) = fit_centroid;

    if ~mod(i,500)
        disp("now is process the " + string(i) +"/" + string(NumData) + "th epoch");
    end
end

ind = find(gan_data.fit_centroid~=0);
gan_data = ExtracSpecialCls(gan_data, ind);

gen_colors = [046,044,105;070,073,156;142,141,200;247,226,219;216,163,152]./255;

%% ground truth & spec curve & hist
figure(1)
hold off
set(gcf, 'Position', [100 100 600 600]); 
% ind = find(simu_data.cls==1);
% average specrum
lambda = linspace(560,659,100);
for i=1:length(gt_data.cls)
    color = gen_colors(i,:);
    plot(lambda,gt_data.spec_value(i,1:100),'LineWidth',2.5,'Color', color)
    title('Averaged Spectrum', 'FontSize', 32, 'FontWeight','bold');
    xlabel('Wavelength [nm]', 'FontSize', 32, 'FontWeight','bold') ;
    ylabel('Intensity', 'FontSize', 32, 'FontWeight','bold') ;
    hold on
    h1 = area(lambda,gt_data.spec_value(i,1:100), 'FaceColor', color, 'FaceAlpha', 0.2);
    set(get(get(h1,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
end
legend(["SC","DSC","DSC311","DS","DC"], 'Location','best')
% legend(["CHCl3","Acetone","C3H8O","EtOH","MeOH"], 'Location','best')
set(gca,'Linewidth',2.5);
set(gca,'FontSize',32);
% 
figure(2)
set(gcf, 'Position', [100 100 600 600]); 
X = categorical({'SC','DSC','DSC311','DS','DC'});
X = reordercats(X,{'SC','DSC','DSC311','DS','DC'});

% X = categorical({'CHCl3','Acetone','C3H8O','EtOH','MeOH'});
% X = reordercats(X,{'CHCl3','Acetone','C3H8O','EtOH','MeOH'});
Y = gt_data.gt_centroid';
b = bar(X, Y, 'FaceColor', gen_colors(1,:));
ylim([580 660])

xtips1 = b(1).XEndPoints;
ytips1 = b(1).YEndPoints;
labels1 = string(roundn(b(1).YData,0));
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom', 'FontSize', 32)

title('Fit Centroid', 'FontSize', 32, 'FontWeight','bold');
xlabel('Label', 'FontSize', 32, 'FontWeight','bold') ;
ylabel('Wavelength [nm]', 'FontSize', 32, 'FontWeight','bold') ;
set(gca,'Linewidth',2.5);
set(gca,'FontSize',32);

%% output & ground truth & raw data 

% 
lipid_cls = ["sc", "dsc", "dsc311", "ds", "dc"];
simu_cls = ["CHCL3","acetone","C3H8O","EtOH","MeOH"];
clsType=["sc","CHCL3","dsc","acetone","EtOH","dsc311","C3H8O","ds","dc","MeOH","noise","fc"];

for i=1:length(lipid_cls)
    clsName = lipid_cls(i);
    cls = find(clsType==clsName)-1;
    
    gan_data_ = ExtracSpecialCls(gan_data, cls);
    raw_data_ = ExtracSpecialCls(raw_data, cls);
    gt_data_ = ExtracSpecialCls(gt_data, cls);
    
    lambda = gan_data_.lambda(21:100);

    figure(2+i)
    set(gcf, 'Position', [100 100 1800 600]); 
    t = tiledlayout(1,3,'TileSpacing','Compact');
    nexttile
    % ind = 1:length(gan_data.id);
    ind = 1:10;
    % average specrum
    y = normalize(mean(gan_data_.spec_value(ind,21:100),1),'range');

    % weibull fitting
%     modelFun =  @(p,x) p(3) .* (x./p(1)).^(p(2)-1) .* exp(-(x./p(1)).^p(2));
%     startingVals = [600 2 1.0];

    % normal fitting
%     modelFun = @(p,x)p(1)*exp(-p(2)*(x-p(3)).^2);
%     startingVals = [1.0, 0.01, 620];

    % double normal fitting
    modelFun=@(p,x)p(1)*exp(-p(2)*(x-p(3)).^2)+p(4)*exp(-p(5)*(x-p(6)).^2);
    startingVals = [1.0, 0.001, 630, 0.5, 0.001, 580];  
    try
        nlModel = fitnlm(lambda,y,modelFun,startingVals);
        % nlModel = fitnlm(lambda,y,modelFun,startingVals);
        fitRMSE = nlModel.RMSE;
        fitValue = normalize(predict(nlModel,lambda'),'range');
    catch
        % disp(clsName+" cannot fit, please check it!")
        fitValue = y;
    end

    % fitValue = y;
    
    gt_y = normalize(gt_data_.spec_value(21:100),'range');
    raw_y = normalize(mean(raw_data_.spec_value(ind,21:100),1),'range');
    plot(lambda, fitValue, 'Color', gen_colors(1,:),'LineWidth',2.5);
    hold on
    plot(lambda, gt_y,  '--', 'Color',  gen_colors(1,:),'LineWidth',2.5);
    plot(lambda, raw_y, 'Color',  gen_colors(5,:),'LineWidth',2.5  );
    hold off
    title('Averaged Spectrum', 'FontSize', 24, 'FontWeight','bold');
    xlabel('Wavelength [nm]', 'FontSize', 24, 'FontWeight','bold') ;
    ylabel('Intensity', 'FontSize', 24, 'FontWeight','bold') ;
    legend('avg. output','ground truth', 'avg. raw', 'Location','south', 'FontSize', 38)
    set(gca,'Linewidth',2.5);
    set(gca,'FontSize',42);
    
    % Centroid Histogram
    nexttile
    hist_range = [625:5:655];
    histogram(gan_data_.raw_centroid(ind), hist_range, 'FaceColor', gen_colors(1,:), 'FaceAlpha',1);
    title('Raw Centroid', 'FontSize', 24, 'FontWeight','bold');
    xlabel('Wavelength [nm]', 'FontSize', 24, 'FontWeight','bold') ;
    ylabel('Count', 'FontSize', 24, 'FontWeight','bold') ;
    set(gca,'Linewidth',2.5);
    set(gca,'FontSize',42);
    
    % Centroid Histogram
    nexttile
    histogram(gan_data_.fit_centroid(ind), hist_range, 'FaceColor', gen_colors(1,:),'FaceAlpha',1);
    gt_centroid = mean(gan_data_.gt_centroid(ind,:),1);
    title('Output Centroid', 'FontSize', 24, 'FontWeight','bold');
    xlabel('Wavelength [nm]', 'FontSize', 24, 'FontWeight','bold') ;
    ylabel('Count', 'FontSize', 24, 'FontWeight','bold') ;

    sgtitle(clsName+' spectrum curve comparison','FontSize', 46, 'FontWeight','bold')

    set(gca,'Linewidth',2.5);
    set(gca,'FontSize',42);
end

%% boxplot

%different type of data
lipid_cls = {'sc', 'dsc', 'dsc311', 'ds', 'dc'};
simu_cls = ["CHCL3","acetone","C3H8O","EtOH","MeOH"];
clsType=["sc","CHCL3","dsc","acetone","EtOH","dsc311","C3H8O","ds","dc","MeOH","noise","fc"];

G_raw = [];
G_fit = [];
group_gan = [];
group_raw = [];

lipid_mean_raw = zeros(1,5);
lipid_mean_gan = zeros(1,5);
lipid_acc_raw = zeros(1,5);
lipid_acc_gan = zeros(1,5);
for i=1:length(lipid_cls)
    clsName = lipid_cls(i);
    cls = find(clsType==clsName)-1;
    
    gan_data_ = ExtracSpecialCls(gan_data, cls);
    raw_data_ = ExtracSpecialCls(raw_data, cls);

    raw_centroid = raw_data_.raw_centroid;
    fit_centroid = gan_data_.fit_centroid;

    G_raw = [G_raw;raw_centroid]; 
    G_fit = [G_fit;fit_centroid];
    group_gan = [group_gan;repmat(clsName,size(fit_centroid,1),1)];
    group_raw = [group_raw;repmat(clsName,size(raw_centroid,1),1)];
    
    disp(clsName)
    disp(mean(G_raw))
    disp(std(G_raw))
    disp("========")
    disp(mean(G_fit))
    disp(std(G_fit))

    lipid_mean_raw(i) = mean(raw_data_.raw_centroid);
    lipid_mean_gan(i) = mean(gan_data_.fit_centroid);
    
    t1 = raw_data_.raw_centroid;
    t2 = gan_data_.fit_centroid;
    g1 = raw_data_.gt_centroid(1);
    lipid_acc_raw(i) = roundn(size(find(t1 > g1-5 &  t1 < g1+5),1) / length(t1),-2);
    lipid_acc_gan(i) = roundn(size(find(t2 > g1-5 &  t2 < g1+5),1) / length(t2),-2);
end

figure(8)
set(gcf, 'Position', [100 100 600 600]); 
pos_raw = [1 2 3 4 5];
hb = boxplot(G_raw, group_raw, 'Positions', pos_raw,'width',.4, 'colors', gen_colors(1,:));
set(hb,'LineWidth',2.5);  
hold on
pos_gan = [1.45 2.45 3.45 4.45 5.45];
hb = boxplot(G_fit, group_gan, 'Positions', pos_gan,'width',.4, 'colors', gen_colors(1,: ));
set(hb,'LineWidth',2.5);  
ylim([560 690])

h = findobj(gca,'Tag','Box');
for j=1:length(h)
    if j < 6
        c = gen_colors(1,:);
    else
        c = gen_colors(5,:);
    end
    patch(get(h(j),'XData'),get(h(j),'YData'),c,'FaceAlpha',0.7);
end

box_vars = findall(gca,'Type','Patch');
legend(box_vars([6,4]), {'outputs','raw'}, 'Location','best', 'FontSize', 32);
% legend('raw','outputs', 'Location','best');
% gt_centroid=gt_data.gt_centroid;
% scatter(pos,gt_centroid, 'MarkerEdgeColor',[70/255 73/255 156/255],...
%               'MarkerFaceColor',[142/255 141/255 200/255],...
%               'LineWidth',1.5);

plot(pos_raw, lipid_mean_raw, '-o', 'LineWidth',2.5, 'HandleVisibility','off','Color',gen_colors(5,:))
plot(pos_gan, lipid_mean_gan, '-o', 'LineWidth',2.5, 'HandleVisibility','off','Color',gen_colors(1,:))
set(gca,'Linewidth',2.5);
set(gca,'FontSize',32);

hold off
xlabel('Label', 'FontSize', 32, 'FontWeight','bold')
ylabel('Wavelength [nm]', 'FontSize', 32, 'FontWeight','bold')

set(gca,'Linewidth',2.5);
set(gca,'FontSize',32);

figure(9)
set(gcf, 'Position', [100 100 600 600]); 
% 
X = categorical({'SC','DSC','DSC311','DS','DC'});
X = reordercats(X,{'SC','DSC','DSC311','DS','DC'});

% X = categorical({'CHCl3','Acetone','C3H8O','EtOH','MeOH'});
% X = reordercats(X,{'CHCl3','Acetone','C3H8O','EtOH','MeOH'});
Y = [lipid_acc_raw; lipid_acc_gan];
b = bar(X, Y,'FaceColor','flat');
ylim([0 1.2])

b(1).CData = gen_colors(1,:);
xtips1 = b(1).XEndPoints;
ytips1 = b(1).YEndPoints;
labels1 = string(b(1).YData);
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom', 'FontSize', 28)

b(2).CData = gen_colors(5,:);
xtips2 = b(2).XEndPoints;
ytips2 = b(2).YEndPoints;
labels2 = string(b(2).YData);
text(xtips2,ytips2,labels2,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom', 'FontSize', 28)

hold off
xlabel('Label', 'FontSize', 32, 'FontWeight','bold') ;
ylabel('Accuracy', 'FontSize', 32, 'FontWeight','bold') ;
set(gca,'Linewidth',2.5);
set(gca,'FontSize',32, 'ytick', [], 'YTickLabel', []);
legend("raw","outputs",'Location','best')

% 
% plot(lipid_acc_raw, 'b-o', 'LineWidth',1.5, 'HandleVisibility','off')
% hold on
% plot(lipid_acc_gan, 'r-o', 'LineWidth',1.5, 'HandleVisibility','off')


%% SR-STORM image
clsName = 'fc';
Left_fileName = strcat(clsName, "/", clsName,"_loc_Int");

LeftFile = strcat(root_path, Left_fileName, ".csv"); 
pos_data = readcsvfile(LeftFile, 500, false);

imgHeight = 257;
imgWidth = 240;
clsType=["sc","CHCL3","dsc","acetone","EtOH","dsc311","C3H8O","ds","dc","MeOH","fc","noise"];
cls = find(clsType==clsName)-1;
gan_data_ = ExtracSpecialCls(gan_data, cls);
raw_data_ = ExtracSpecialCls(raw_data, cls);

SpecHistorm = zeros([imgHeight imgWidth]);
gan_sr_curves = zeros([imgHeight imgWidth 120]);
raw_sr_curves = zeros([imgHeight imgWidth 120]);

Y = round(gan_data_.s_px);
X = round(gan_data_.s_py);
% SR-STORM Image

for i=1:length(X)
    gan_sr_curves(X(i), Y(i),:) = squeeze(gan_sr_curves(X(i), Y(i),:))' + gan_data_.spec_value(i,:);
    raw_sr_curves(X(i), Y(i),:) = squeeze(raw_sr_curves(X(i), Y(i),:))'+ raw_data_.spec_value(i,:);
    SpecHistorm(X(i), Y(i)) = SpecHistorm(X(i), Y(i)) + 1;
end

SpecHistorm(SpecHistorm==0) = [1];
gan_sr_curves = gan_sr_curves ./ SpecHistorm;
raw_sr_curves = raw_sr_curves ./ SpecHistorm;

lambda = gan_data_.lambda;
[M,gan_I] = max(gan_sr_curves,[],3);
gan_sr_image = lambda(gan_I);
[M,raw_I] = max(raw_sr_curves,[],3);
raw_sr_image = lambda(raw_I);

% STORM Image
StormImg = zeros([imgHeight imgWidth]);
Y = round(pos_data.x);
X = round(pos_data.y);
for i=1:length(X)
    StormImg(X(i), Y(i)) = StormImg(X(i), Y(i)) + 1;
end

figure(12)
set(gcf, 'Position', [100 100 600 600]); 
hist_range = [1:2:30];
histogram(StormImg, hist_range, 'FaceColor', gen_colors(1,:), 'FaceAlpha',1);
xlabel('Counts', 'FontSize', 24, 'FontWeight','bold') ;
set(gca,'Linewidth',2.5);
set(gca,'FontSize',32, 'FontWeight','bold');



StormImg = (StormImg - min(min(StormImg))) / (max(max(StormImg)) - min(min(StormImg)));
ImageAdj = imadjust(StormImg, stretchlim(StormImg), []);

lowerLim = 560;
upperLim = 648;

figure(9);
t = tiledlayout(1,2,'TileSpacing','Compact');
nexttile
set(gcf, 'Position', [100 100 1200 600]); 
raw_sr_image(raw_sr_image < lowerLim) = lowerLim; % adjust contrast
raw_sr_image(raw_sr_image > upperLim) = upperLim; % by cutoff and saturation
imgRatioColor = ind2rgb(im2uint8(mat2gray(raw_sr_image)),jet(256));

imgRatioColorMask =  imgRatioColor .* ImageAdj;

imshow(imgRatioColorMask)
colormap jet;
colorbar; caxis([lowerLim,upperLim]);
title("raw SR-STORM", 'FontSize',28)

set(gca,'Linewidth',2.5);
set(gca,'FontSize',32);

nexttile
gan_sr_image(gan_sr_image < lowerLim) = lowerLim; % adjust contrast
gan_sr_image(gan_sr_image > upperLim) = upperLim; % by cutoff and saturation
imgRatioColor = ind2rgb(im2uint8(mat2gray(gan_sr_image)),jet(256));

imgRatioColorMask =  imgRatioColor .* ImageAdj;

imshow(imgRatioColorMask)
colormap jet;
colorbar; caxis([lowerLim,upperLim]);
title("GAN output", 'FontSize',28)

set(gca,'Linewidth',2.5);
set(gca,'FontSize',32);

figure(10)
set(gcf, 'Position', [100 100 600 600]); 
% ax3=subplot(1,3,3);
imshow(ImageAdj, [])
% title("STROM", 'FontSize',20)

set(gca,'Linewidth',2.5);
set(gca,'FontSize',10);

coor = input("pleas input the centroid of paritcle(input format [a b]):");
xi = coor(2);
yi = coor(1);
% rectangle(ax1, 'Position',[yi-2 xi-2 4 4],'LineWidth',2,'EdgeColor','c');
% rectangle(ax2, 'Position',[yi-2 xi-2 4 4],'LineWidth',2,'EdgeColor','c');
% rectangle(ax3, 'Position',[yi-2 xi-2 4 4],'LineWidth',2,'EdgeColor','c');

figure(11)
set(gcf, 'Position', [100 100 600 600]); 
t = tiledlayout(1,2,'TileSpacing','Compact');

gan_px_cruves = normalize(squeeze(gan_sr_curves(xi, yi,21:100))','range');
raw_px_cruves = normalize(squeeze(raw_sr_curves(xi, yi,21:100))','range');

plot(lambda(21:100), raw_px_cruves,'--', 'Color',gen_colors(1,:), 'LineWidth',3.5)
hold on
plot(lambda(21:100), gan_px_cruves, 'Color',gen_colors(1,:), 'LineWidth',3.5);
title('Averaged Spectrum', 'FontSize', 20, 'FontWeight','bold');
xlabel('Wavelength [nm]', 'FontSize', 20, 'FontWeight','bold') ;
ylabel('Intensity', 'FontSize', 20, 'FontWeight','bold') ;
legend('avg. raw','avg. GAN', 'Location','best')
set(gca,'Linewidth',2.5);
set(gca,'FontSize',32);
hold off;

% nexttile
% plot(lambda(21:100), gan_px_cruves, 'r-', 'LineWidth',3.5);
% title('Averaged Spectrum', 'FontSize', 20, 'FontWeight','bold');
% xlabel('Wavelength [nm]', 'FontSize', 20, 'FontWeight','bold') ;
% ylabel('Fluoresence Intensity', 'FontSize', 20, 'FontWeight','bold') ;
% legend('avg. gan', 'Location','best')
% set(gca,'Linewidth',2.5);
% set(gca,'FontSize',32);



%% 
figure(11)
raw_data_ = ExtracSpecialCls(raw_data, 3);
gt_data_ = ExtracSpecialCls(gt_data, 3);

gt_y = normalize(gt_data_.spec_value(21:5:100),'range');
raw_1 = normalize(mean(raw_data_.spec_value(1,21:5:100),1),'range');
raw_2 = normalize(mean(raw_data_.spec_value(1:100,21:5:100),1),'range');
raw_3 = normalize(mean(raw_data_.spec_value(1:500,21:5:100),1),'range');


lambda = gt_data_.lambda(21:5:100);

plot(lambda, raw_1,'r-', lambda, raw_2, 'b-', lambda, raw_3, 'g-', lambda, gt_y, '--', 'LineWidth',2.5);
xlabel('Wavelength [nm]', 'FontSize', 24, 'FontWeight','bold') ;
ylabel('Intensity', 'FontSize', 24, 'FontWeight','bold') ; 
legend('single molecule','100 molecules', '500 molecules', '10,000 molecules', 'Location','best')
set(gca,'Linewidth',2.5);
set(gca,'FontSize',28);
set(gca,'YTick',[0:0.5:1]);
set(gca,'XTick',[580:20:680]);
box off;


%% VMD

raw_data_ = ExtracSpecialCls(raw_data, 3);
gt_data_ = ExtracSpecialCls(gt_data, 3);
raw_1 = normalize(mean(raw_data_.spec_value(1,21:100),1),'range');
lambda = gt_data_.lambda(21:5:100);

% VMD分解参数设置
alpha=2000;  % alpha   - 惩罚因子
tol=1e-7;    % tol     - 收敛容差，是优化的停止准则之一，可以取 1e-6~5e-6
K=4;         % K       - 指定分解模态数
type = 2;    % 采用第三方vmd函数进行分解

% 以下输入参数在使用MATLAB内置函数的时候不需要输入（可以置为nan）
tau=0;      % tau     - time-step of the dual ascent ( pick 0 for noise-slack )
DC=1;       % DC      - true if the first mode is put and kept at DC (0-freq)
init=1;     % init    - 0 = all omegas start at 0
imf = pVMD(raw_1,80, alpha, K, tol, type, tau, DC, init); %p文件可调用，无法查看源码，需要源码请获取完整版代码
close all

res_1 = normalize(imf(4,:),'range');  %residual
imf_1 = normalize(imf(1,:),'range'); 
imf_2 = normalize(imf(2,:),'range');
imf_3 = normalize(imf(3,:),'range');

h1 = figure(12);
plot(lambda, raw_1(1:5:end),'r-', 'LineWidth',2.5);
set(gca,'Linewidth',3.5);
set(gca,'FontSize',36);
set(gca,'YTick',[0:0.5:1]);
set(gca,'XTick',[580:20:680]);
xlabel('Wavelength [nm]', 'FontSize', 32, 'FontWeight','bold') ;
ylabel('Intensity', 'FontSize', 32, 'FontWeight','bold') ;
box off;

figure(13)
plot(lambda, res_1(1:5:end),'r-', 'LineWidth',2.5);
set(gca,'Linewidth',3.5);
set(gca,'FontSize',36);
set(gca,'YTick',[0:0.5:1]);
set(gca,'XTick',[580:20:680]);
xlabel('Wavelength [nm]', 'FontSize', 32, 'FontWeight','bold') ;
ylabel('Intensity', 'FontSize', 32, 'FontWeight','bold') ;
box off;

figure(14)
lambda = gt_data_.lambda(21:100);
set(gcf, 'Position', [100 100 600 600]); 
subplot(5,1,1) 
plot(lambda, raw_1, 'r-', 'LineWidth', 2.5);
set(gca,'Linewidth',3.5);
set(gca,'FontSize',22);
set(gca, 'XtickLabel', [])
ylabel('raw', 'FontSize', 22, 'FontWeight','bold') ;
%box off
grid on

subplot(5,1,2)
plot(lambda, imf_1,'r-', 'LineWidth',2.5);
set(gca,'Linewidth',3.5);
set(gca,'FontSize',22);
set(gca, 'XtickLabel', [])
ylabel('IMF1', 'FontSize', 22, 'FontWeight','bold') ;
%box off
grid on

subplot(5,1,3)
plot(lambda, imf_2,'r-', 'LineWidth',2.5);
set(gca,'Linewidth',3.5);
set(gca,'FontSize',22);
set(gca, 'XtickLabel', [])
ylabel('IMF2', 'FontSize', 22, 'FontWeight','bold') ;
%box off
grid on

subplot(5,1,4)
plot(lambda, imf_3,'r-', 'LineWidth',2.5);
set(gca,'Linewidth',3.5);
set(gca,'FontSize',22);
set(gca, 'XtickLabel', [])
ylabel('IMF3', 'FontSize', 22, 'FontWeight','bold') ;
%box off
grid on

subplot(5,1,5)
plot(lambda, res_1,'r-', 'LineWidth',2.5);
set(gca,'Linewidth',3.5);
set(gca,'FontSize',22);
ylabel('res', 'FontSize', 22, 'FontWeight','bold') ;
xlabel('Wavelength [nm]', 'FontSize', 22, 'FontWeight','bold') ;
grid on


%% single molecule spectrum&ground truth & gan_ouput

figure(14)
raw_data_ = ExtracSpecialCls(raw_data, 3);
gt_data_ = ExtracSpecialCls(gt_data, 3);
gan_data_ = ExtracSpecialCls(gan_data, 3);

gt_y = normalize(gt_data_.spec_value(21:5:100),'range');
raw_1 = normalize(mean(raw_data_.spec_value(1,21:5:100),1),'range');
raw_2 = normalize(mean(raw_data_.spec_value(2,21:5:100),1),'range');
raw_3 = normalize(mean(raw_data_.spec_value(3,21:5:100),1),'range');
gan_y = normalize(mean(gan_data_.spec_value(1,21:5:100),1),'range');


lambda = gt_data_.lambda(21:5:100);

plot(lambda, raw_1,'r-', lambda, raw_2, 'b-', lambda, raw_3, 'g-', 'LineWidth',2.5);
legend('molecule 1','molecule 2', 'molecule 3', 'Location','best')
set(gca,'Linewidth',3.5);
set(gca,'FontSize',36);
set(gca,'YTick',[0:0.5:1]);
set(gca,'XTick',[580:20:680]);
xlabel('Wavelength [nm]', 'FontSize', 32, 'FontWeight','bold') ;
ylabel('Intensity', 'FontSize', 32, 'FontWeight','bold') ;
box off;

figure(15)
plot(lambda, gt_y, 'LineWidth',2.5);
set(gca,'Linewidth',3.5);
set(gca,'FontSize',36);
set(gca,'YTick',[0:0.5:1]);
set(gca,'XTick',[580:20:680]);
xlabel('Wavelength [nm]', 'FontSize', 32, 'FontWeight','bold') ;
ylabel('Intensity', 'FontSize', 32, 'FontWeight','bold') ;
box off;

figure(16)
plot(lambda, gan_y, 'r--', 'LineWidth',2.5);
set(gca,'Linewidth',3.5);
set(gca,'FontSize',36);
set(gca,'YTick',[0:0.5:1]);
set(gca,'XTick',[580:20:680]);
xlabel('Wavelength [nm]', 'FontSize', 32, 'FontWeight','bold') ;
ylabel('Intensity', 'FontSize', 32, 'FontWeight','bold') ;
box off;

%% calibrated curve
pixel_shift = -[19.8854897, 11.93918715, 0, -13.9645882];
lambda_val = [532, 561, 591, 638];

f_sp=@(x)(0.002661*x.^3+0.02754*x.^2+ 2.462*x+591); % 3-rd poly fitting
x = -20:1:20;
fit_y = f_sp(x);
figure(17)
plot(pixel_shift, lambda_val, 'o', 'MarkerFaceColor',gen_colors(1,:),...
    'MarkerEdgeColor','k',...
    'MarkerSize',6)
hold on
plot(x, fit_y,'r-','Color',gen_colors(1,:), 'LineWidth',2.5);
hold off
set(gca,'Linewidth',2.5);
set(gca,'FontSize',24);
set(gca,'YTick',[520:50:680]);
set(gca,'XTick',[-20:5:20]);
xlabel('pixel shift', 'FontSize', 32, 'FontWeight','bold') ;
ylabel('Wavelength [nm]', 'FontSize', 32, 'FontWeight','bold') ;
box off;



