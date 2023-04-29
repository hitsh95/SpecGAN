function f = readcsvfile(filename, tInt, bImageJ)

% 1       2     3   4     5        6          7       8      9           10  
% id	frame	x	y	sigma	intensity	offset	bkgstd	chi2	uncertainty


% filename='testRead.bin';
fid = fopen(filename,'r');

A = textscan(fid, '%d%d%f%f%f%f%f%f%f%f', 'Delimiter', ',', 'HeaderLines', 1);
% Read float values first
id = cell2mat(A(1));
frame = cell2mat(A(2));

if bImageJ == true
    x = round(cell2mat(A(3)) / 100 + 1);
    y = round(cell2mat(A(4)) / 100 + 1);
else 
    x = round(cell2mat(A(3)) / 100);
    y = round(cell2mat(A(4)) / 100);
end

sigma = cell2mat(A(5));
I = cell2mat(A(6));
offset = cell2mat(A(7));
bg = cell2mat(A(8));
chi2 = cell2mat(A(9));
uncertainty = cell2mat(A(10));

clear A;
fclose(fid);

ind = find(uncertainty<20 & I>tInt);%&I>35000
% ind = id;
mol.id = id(ind);
mol.frame = frame(ind);
mol.x = x(ind);
mol.y = y(ind);
mol.sigma = sigma(ind);
mol.offset = offset(ind);
mol.bg = bg(ind);
mol.I = I(ind);
mol.bg = bg(ind);
mol.chi2 = chi2(ind);
mol.uncertainty = uncertainty(ind);

f = mol;




