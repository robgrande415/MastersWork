for i = 1:size(NBHT,1)
    NBHT{i,1}.mae = 0;
    NBHT{i,1}.kl_tol = kl_tol_settings(i);
    NBHT{i,1}.detect_size = detect_settings(i);
    for j = 1:size(NBHT,2)
        NBHT{i,1}.mae = NBHT{i,1}.mae + mean(abs(NBHT{i,j}.mean - NBHT{i,j}.f))/size(NBHT,2);
    end
end

%%
n = sqrt(size(NBHT,1));
hm = zeros(n,n);
xlabels = zeros(1,n);
ylabels = zeros(1,n);
for i = 1:size(NBHT,1)
    hm(mod(i,n)+1,floor(i/n)+1) = NBHT{i,1}.mae;
    xlabels(floor(i/n)+1) = NBHT{i,1}.detect_size;
    ylabels(mod(i,n)+1) = NBHT{i,1}.kl_tol;
end
hm = hm(:,1:15)
hm = hm(2:17,:)
ylabels = ylabels(2:17)
xlabels = xlabels(1:15)
%%
hmo = HeatMap(hm,'RowLabels',ylabels,'ColumnLabels',xlabels,'Colormap','jet','DisplayRange',0.2)
addTitle(hmo,'Mean Abs. Error')
addXLabel(hmo,'m')
addYLabel(hmo,'\eta')