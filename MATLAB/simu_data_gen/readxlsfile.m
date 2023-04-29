function f = readxlsfile(filename)

T=readcell(filename);

label = string(T(1,1:3:end));
lambda = cell2mat(T(3:end,1));
val = cell2mat(T(3:end,2:3:end));
norm_val = normalize(val,1,"range");

% cls=["d","ds31","ds21","dsc311","ds","dsc211","ds12","dsc","ds13","dsc112","dsc121","dsc113","dsc131"];
% cls=["d","ds31","ds21","dsc311","ds","ds12","ds13","dsc211","dsc112","dsc113","dsc","dsc131","dsc121"];
cls=["sc","CHCL3","dsc","acetone","EtOH","dsc311","C3H8O","ds","d","MeOH"];
% 蓝移->红移

mol.lambda = lambda;
mol.val = norm_val;
c = zeros(size(label));
for i=1:length(label)
    c(i)=find(cls==label(i))-1;
end
mol.cls = c;
mol.clsName = label;

f = mol;
end

