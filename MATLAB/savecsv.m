function savecsv(root_path, filename, data, Index, lambda, datamode)
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
output = strcat(root_path, filename);
csvtitle = {'id', 'cls', 'clsName', 'ku', 'cv', 'raw_centroid', 'gt_centroid', ...
    's_px', 's_py'};
spec_lambda = string(lambda);

data.id = data.id(Index);
data.cls = data.cls(Index);
data.clsName = data.clsName(Index);
data.ku = data.ku(Index);
data.cv = data.cv(Index);
data.gt_centroid = data.gt_centroid(Index);
data.raw_centroid = data.raw_centroid(Index);
data.s_px = data.s_px(Index);
data.s_py = data.s_py(Index);
data.spec_value = data.spec_value(Index,:);
% data.res = data.res(Index,:);
data.norm_spec_value = data.norm_spec_value(Index,:);


spec_table = table(data.id, data.cls, data.clsName,  ...
    data.ku, data.cv, data.raw_centroid, data.gt_centroid, data.s_px, data.s_py,...
'VariableNames', csvtitle);

if datamode == "raw"
    for i=1:length(lambda)
        tab = table(data.norm_spec_value(:,i), 'VariableNames', spec_lambda(i));
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

