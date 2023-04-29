infile='PointPairs/ControlPointPairs_1114.txt';
outfile = 'STORM_0001_warp.mat';

PairData = importdata(infile,'\t')+1; % matlab data start from (1,1), whereas the imageJ from (0,0)
input = PairData(:,1:2);

base=PairData(:,3:4);
baseX=base(:,1);
baseY=base(:,2);

figure(1)
plot(input(:,1),input(:,2),'k.',baseX,baseY,'m.');

tform = cp2tform(input,base,'projective');
[tx,ty] = tforminv(tform,baseX,baseY);

figure(2)
plot(input(:,1),input(:,2),'k.',tx,ty,'m.');

% Saving tform 
save(outfile,'tform');

