function savecsv(root_path, filename, data, Index, lambda, datamode)
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
output = strcat(root_path, filename);
csvtitle = {'id', 'cls', 'clsName', 'ku', 'cv', 'raw_centroid', 'gt_centroid', 'noise_mu', 'noise_sigma', 'noiseType'};

spec_lambda = string(lambda);

data.id = data.id(Index);
data.cls = data.cls(Index);
data.clsName = data.clsName(Index);
data.ku = data.ku(Index);
data.cv = data.cv(Index);
data.sigma = data.sigma(Index);
data.gt_centroid = data.gt_centroid(Index);
data.raw_centroid = data.raw_centroid(Index);
data.noise_mu = data.noise_mu(Index);
data.noise_sigma = data.noise_sigma(Index);
data.noiseType = data.noiseType(Index);
data.spec_value = data.spec_value(Index,:);
data.res = data.res(Index,:);

spec_table = table(data.id, data.cls, data.clsName, ...
    data.ku, data.cv, data.raw_centroid, data.gt_centroid, ...
    data.noise_mu, data.noise_sigma, data.noiseType, ...
'VariableNames', csvtitle);

if datamode == "raw"
    for i=1:length(lambda)
        tab = table(data.spec_value(:,i), 'VariableNames', spec_lambda(i));
        spec_table = [spec_table tab];
    end

elseif datamode == "res"
    for i=1:length(lambda)
        tab = table(data.res(:,i), 'VariableNames', spec_lambda(i));
        spec_table = [spec_table tab];
    end
end

writetable(spec_table, output);
end
