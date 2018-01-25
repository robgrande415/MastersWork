function [ s_prime] = glider_trans( s,a,params )
%GLIDER_TRANS Summary of this function goes here
%   Detailed explanation goes here

a_set = {'fl', 'ff', 'fr'};
[s_prime, si] = hex_move(s, a_set{a});

params.origSprime = s_prime;
params.origSi = si;

s_prime = min([grid_sizex, grid_sizey, n_headings], max([1, 1, 1], s_prime));
params.s_prime = s_prime;
params.s_primen = (s_prime(3)-1)*num_cells + (s_prime(2)-1)*grid_sizex + s_prime(1);
    
params.si = min([grid_sizex, grid_sizey, n_headings], max([1, 1, 1], si));
params.si_n = (si(:,3)-1)*num_cells + (si(:,2)-1)*grid_sizex + si(:,1);
    


end

