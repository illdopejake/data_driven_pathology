function mi4d_ecdf = voxelwise_pet_prob_yasser2016(mi4d, refz) 

cut_inf = prctile(refz, 5)
cut_sup = prctile(refz, 95)
[MAXs] = bootstrp(20000, @max, refz(refz >= cut_inf & refz <= cut_sup)); 
parmhat = evfit(double(MAXs));
mi4d_ecdf = evcdf(mi4d, parmhat(1), parmhat(2));