clear all;
warning off

root_path = '../../data/lipid_data/sc/';
filename = 'sc';
output = strcat(root_path, filename, "_spec.csv");
LeftFile = strcat(root_path, filename, "_loc_Int.csv"); 
Leftimg = strcat(root_path, filename, "_left.tif");
Rightimg = strcat(root_path, filename, "_right.tif");

Left = readcsvfile(LeftFile, 1500, false);

%% read tiff file
fprintf(1,'\nLoading Image of Position...\n');

imgInfo = imfinfo(Leftimg);
imgHeight = imgInfo(1).Height;
imgWidth = imgInfo(1).Width;
imgDepth = length(imgInfo);
imgBitDepth = ['uint',num2str(imgInfo(1).BitDepth)];
ImStack = zeros(imgHeight, imgWidth, imgDepth,imgBitDepth);
for ii = 1 : length(imgInfo)
    ImStack(:,:,ii) = imread(Leftimg, ii);
end
fprintf(1, 'done!')

figure(1)
tiledlayout(1,2,'TileSpacing','Compact');
nexttile
imshow(ImStack(:,:,1),[]);
title('position', 'FontSize', 16, 'FontWeight','bold')
hold on;
p_f = find(Left.frame==1);
plot(Left.x(p_f), Left.y(p_f), 'o');

%% 
%Warping the data of the second objective
warpfile = 'STORM_0001_warp.mat'; %This was generated in the first step of mapping.
load(warpfile);
[tx,ty] = tforminv(tform, Left.x, Left.y);

fprintf(1,'\nLoading Image of Spectrum...\n');
imgInfo = imfinfo(Rightimg);
imgSpecRow = imgInfo(1).Height;
imgSpecCol = imgInfo(1).Width;
imgDepth = length(imgInfo);
imgBitDepth = ['uint',num2str(imgInfo(1).BitDepth)];
SpecStack = zeros(imgSpecRow, imgSpecCol, imgDepth,imgBitDepth);
for ii = 1 : length(imgInfo)
    SpecStack(:,:,ii) = imread(Rightimg, ii);
end
fprintf(1, 'done!\n')

nexttile
imshow(SpecStack(:,:,1), []);
title('591nm', 'FontSize', 16, 'FontWeight','bold')
hold on;
plot(tx(p_f), ty(p_f), 'o');
fprintf(1,'Data of Spectrum warped!\n');

%% calc spectrum 
DisplayFrame=500;
NowFrameNum=-1000;
SpecHeight = 3;
SpecWidth = 40;
id = 0;
NumLeft = size(Left.x,1);

cls=["sc","CHCL3","dsc","acetone","EtOH","dsc311","C3H8O","ds","dc","MeOH","noise","fc"];

mol = struct( 'id', zeros(NumLeft,1),...
    'frame', zeros(NumLeft,1), ...
    'x_t', Inf(NumLeft,1), ...
    'y_t', Inf(NumLeft,1), ...
    'px', Inf(NumLeft,1), ...
    'py', Inf(NumLeft,1), ...
    'spectrum',repmat(double(0), NumLeft, 120), ...
    'centroid',zeros(NumLeft,1), ...
    'spec_photons', zeros(NumLeft,1), ...
    'spec_bk_photons', zeros(NumLeft,1), ...
    'spec_uncertainty',  zeros(NumLeft,1), ...
    'sigma',  zeros(NumLeft,1), ...
    'ku', zeros(NumLeft,1), ...
    'cv', zeros(NumLeft,1), ...
    'fitSkew', zeros(NumLeft,1), ...
    'cls', Inf(NumLeft,1), ...
    'clsName', string(zeros(NumLeft,1)));

spec_data.lambda = linspace(560, 679,120);
% lambda-pixel mapping
f_sp=@(x)(0.002661*x.^3+0.02754*x.^2+ 2.462*x+591); % 3-rd poly fitting
% f_sp = @(x)(0.03019*x.^2-3.078*x+593.7);
% f_sp = @(x)(x);
pix_shift = 5; % middle_point of spectrum:  591->0;531->-20;605->5
pix = 1:SpecWidth;
lambda = f_sp(SpecWidth/2 - pix + pix_shift);

pp = [];

for i=1:NumLeft
    CurrentFrameNum = Left.frame(i);    
    id = id + 1;

    if CurrentFrameNum~=NowFrameNum
        NowFrameNum=CurrentFrameNum;  
        pp = find(Left.frame==CurrentFrameNum);
    end
    
    x_t = tx(i);
    y_t = ty(i);
    px = Left.x(i);
    py = Left.y(i);

%         if px < 15 || px > 105 || py < 50 || py > 150
%             continue
%         end
        
    % remove the overlap spectrum
    o_ix = find(abs(x_t-tx(pp))<SpecWidth);
    o_iy = find(abs(y_t-ty(pp))<SpecHeight);

    o_ind = intersect(o_ix,o_iy);
    o_ind(o_ind==(id - pp(1)+1)) = [];

    if ~isempty(o_ind)
        mol.id(id) = 0;
        continue
    end

    RspecX = tx(i) - floor(SpecWidth/2) - pix_shift;
    RspecY = ty(i) - floor(SpecHeight/2);

    ImgRight = SpecStack(:,:,CurrentFrameNum);
    if RspecX + SpecWidth > (size(ImgRight,2)-1) || RspecX < 1 || RspecY + SpecHeight > (size(ImgRight,1)-1) || RspecY < 1
        continue
    end

    spec = mean(ImgRight(RspecY:RspecY+SpecHeight-1, RspecX:RspecX+SpecWidth-1), 1);
    
    [pks,locs] = max(spec, spec_data.lambda);

    % uncertainty
    uncertainty = (max(spec) - min(spec)) / mean(fitValue);

    % kurtosis
    fitku = kurtosis(spec);

    % sigma
    fitSigma = std(spec);

    % Coefficient of Variation
    fitCv = fitSigma / mean(spec);
    fitSkew = max(spec)/mean(spec);
% 
%     figure(5)
%     plot(lambda, spec, 'o', locs,pks, '+');
%     line(lambda, spec);
%     xlabel('lambda');
%     ylabel('Int');
%     title(['No.', num2str(id), ' single Spectrum'], 'FontSize', 14, 'FontWeight','bold');
%     line(spec_data.lambda', predict(nlModel,spec_data.lambda'), 'Color', 'r');

    % pks = [600];

    % photons
    [photons,maxPos] = max(pks);
    centroid = locs(maxPos);

    if fitCv<0.1 || isempty(pks) || fitku>25 || fitSkew>3%%|| fitRMSE>4e3 || photons>1500
        mol.id(id) = 0;
        continue
    end

    % photons = mean(maxk(spec, 5));

    mol.id(id) = id;
    mol.frame(id) = CurrentFrameNum;
    y = interp1(lambda, spec, spec_data.lambda, 'linear');
    mol.spectrum(id,:) = y;
    mol.x_t(id) = x_t;
    mol.y_t(id) = y_t;
    mol.px(id) = px;
    mol.py(id) = py;
    mol.centroid(id) = centroid;
    mol.sigma(id) = fitSigma;
    mol.spec_photons(id) = photons;
    mol.fitSkew(id) = fitSkew;
    mol.ku(id) = fitku;
    mol.cv(id) = fitCv;
    mol.cls(id) = find(cls==filename)-1;
    mol.clsName(id) = string(filename);

end

ind = find(mol.id ~= 0);

spec_data.id = mol.id(ind);
spec_data.frame = mol.frame(ind);
spec_data.x_t = mol.x_t(ind);
spec_data.y_t = mol.y_t(ind);
spec_data.px = mol.px(ind);
spec_data.py = mol.py(ind);
spec_data.spectrum = mol.spectrum(ind,:);
spec_data.centroid = mol.centroid(ind);
spec_data.sigma = mol.sigma(ind);
spec_data.spec_photons = mol.spec_photons(ind);
spec_data.ku = mol.ku(ind);
spec_data.cv = mol.cv(ind);
spec_data.cls = mol.cls(ind);
spec_data.clsName = mol.clsName(ind);


%% plot averaged spectrum 
figure(3)
set(gcf, 'Position', [100 100 900 600]); 
t = tiledlayout(2,3,'TileSpacing','Compact');

% average specrum
nexttile
y = mean(spec_data.spectrum,1);
spec_data.avg_spectrum = y;
plot(spec_data.lambda, y, 'r-','LineWidth',1.2);
title('Averaged Spectrum', 'FontSize', 14, 'FontWeight','bold');
xlabel('Wavelength [nm]', 'FontSize', 14, 'FontWeight','bold') ;
ylabel('Fluoresence Intensity', 'FontSize', 14, 'FontWeight','bold') ;

% Centroid Histogram
nexttile
histogram(spec_data.centroid);
title('Centroid Histogram', 'FontSize', 14, 'FontWeight','bold');
xlabel('Wavelength [nm]', 'FontSize', 14, 'FontWeight','bold') ;
ylabel('Count', 'FontSize', 14, 'FontWeight','bold') ;

% Centroid Vs Photons
nexttile
scatter(spec_data.centroid, spec_data.spec_photons, '.');
title('Centroid Vs Photons', 'FontSize', 14, 'FontWeight','bold');
xlabel('Centroid', 'FontSize', 14, 'FontWeight','bold') ;
ylabel('Photons', 'FontSize', 14, 'FontWeight','bold') ;

% Centroid Vs Sigma
nexttile
scatter(spec_data.centroid, spec_data.sigma, '.');
title('Centroid Vs Sigma', 'FontSize', 14, 'FontWeight','bold');
xlabel('Centroid', 'FontSize', 14, 'FontWeight','bold') ;
ylabel('Sigma', 'FontSize', 14, 'FontWeight','bold') ;

% Centroid Vs kurtosis
nexttile
scatter(spec_data.centroid, spec_data.ku, '.');
title('Centroid Vs kurtosis', 'FontSize', 14, 'FontWeight','bold');
xlabel('Centroid', 'FontSize', 14, 'FontWeight','bold') ;
ylabel('kurtosis', 'FontSize', 14, 'FontWeight','bold') ;

% Centroid Vs Variation Coefficient
nexttile
scatter(spec_data.centroid, spec_data.cv, '.');
title('Centroid Vs CV', 'FontSize', 14, 'FontWeight','bold');
xlabel('Centroid', 'FontSize', 14, 'FontWeight','bold') ;
ylabel('Coefficient of Variation', 'FontSize', 14, 'FontWeight','bold') ;

%% plot spectrum mapping image
Y = round(spec_data.px);
X = round(spec_data.py);
% SR-STORM Image
SpecHistorm = zeros([imgHeight imgWidth]);
SrStromImg = zeros([imgHeight imgWidth]);
for i=1:length(X)
    SrStromImg(X(i), Y(i)) = SrStromImg(X(i), Y(i)) + spec_data.centroid(i);
    SpecHistorm(X(i), Y(i)) = SpecHistorm(X(i), Y(i)) + 1;
end

SpecHistorm(SpecHistorm==0) = [1];
SrStromImg = SrStromImg ./ SpecHistorm;

% STORM Image
StormImg = zeros([imgHeight imgWidth]);
Y = round(Left.x);
X = round(Left.y);
for i=1:length(X)
    StormImg(X(i), Y(i)) = StormImg(X(i), Y(i)) + 1;
end


%%
lowerLim = 612;
upperLim = 648;
figure(4)
subplot(1,3,1)
imshow(SrStromImg);axis image;
colormap jet;
colorbar; caxis([lowerLim,upperLim]);
title("Spectrum historm image")
% saveas(gcf,'imgRatioWithColorbar.tif');

subplot(1,3,2)
SrStromImg(SrStromImg < lowerLim) = lowerLim; % adjust contrast
SrStromImg(SrStromImg > upperLim) = upperLim; % by cutoff and saturation
imgRatioColor = ind2rgb(im2uint8(mat2gray(SrStromImg)),jet(256));

StormImg = (StormImg - min(min(StormImg))) / (max(max(StormImg)) - min(min(StormImg)));
ImageAdj = imadjust(StormImg, stretchlim(StormImg), []);
imgRatioColorMask =  imgRatioColor .* ImageAdj;

imshow(imgRatioColorMask)
colormap jet;
colorbar; caxis([lowerLim,upperLim]);
title("SR-STORM reconstruction image")

subplot(1,3,3)
imshow(ImageAdj, [])
title("Single Particle historm image")


%% save to csv file

% 红移->蓝移
label=find(cls==filename)-1;
spec_lambda = string(spec_data.lambda);

% ground truth data
gt_filename = strcat(root_path, filename, '_gt.csv');
csvtitle = {'cls', 'clsName', 'gt_centroid'};

norm_spec = normalize(spec_data.avg_spectrum, 2, 'range');

[photons,maxPos] = max(norm_spec);
gt_centroid = spec_data.lambda(maxPos);

spec_table = table(label, string(filename), gt_centroid', 'VariableNames', csvtitle);
for i=1:length(spec_lambda)
    tab = table(norm_spec(i), 'VariableNames', spec_lambda(i));
    spec_table = [spec_table tab];
end

writetable(spec_table, gt_filename);

%%%%%%%%%%%%%%%%%
csvtitle = {'id', 'frame', 'cls', 'clsName', 's_px', 's_py', 'sigma', ...
    'raw_centroid', 'gt_centroid','ku', 'cv', 'photons'};
spec_data.gt_centroid = ones(size(spec_data.id))*gt_centroid;

spec_table = table(spec_data.id, spec_data.frame, spec_data.cls, spec_data.clsName,spec_data.px, spec_data.py, ...
    spec_data.sigma, spec_data.centroid, spec_data.gt_centroid, spec_data.ku, spec_data.cv, spec_data.spec_photons, ...
'VariableNames', csvtitle);


norm_spec_data = normalize(spec_data.spectrum, 2, 'range');
for i=1:length(spec_lambda)
    tab = table(spec_data.spectrum(:,i), 'VariableNames', spec_lambda(i));
    spec_table = [spec_table tab];
end
writetable(spec_table, output);


