function [rew, breaker ] = glider_rew( s,a, params )
%GLIDER_REW Summary of this function goes here
%   Detailed explanation goes here

    origSprime = params.origSprime;
    s_primen = params.s_primen;
    origSi = params.origSi;
    si_n = params.si_n;

    breaker= false;
    grid_sizex = 20;                    % grid size
grid_sizey = 20;                    % grid size
num_cells = grid_sizex*grid_sizey ; % total number of cells
n_headings = 6;                     % discrete number of available state headings

base_speed = 15;                    % platform velocity
base_mass = 5.44;                   % platform mass
L_on_D = 20;                        % lift to drag ratio

delta_t = 1;                        % timestep
grid_spacing = base_speed*delta_t;  % define grid spacing by platform dynamics
n_timesteps = 1e3;                  % number of timesteps

g = 9.81;                           % gravity


%% Generate hexagonal grid
[X Y] = meshgrid(1:grid_sizex, 1:grid_sizey);
[XH, YH] = square2hex(X, Y);


%% Thermals
n_sources = 1;
source_strength = [4] ;         % [nx1] where n = n_sources
source_radius = [3] ;           % [nx1] where n = n_sources
source_locations = [17 12] ;    % [nx2] where n = n_sources


%% Shear
shear_strength = 0.5;
shear_limits = [5, 15; 10, 20]'; % area containing shear region
shear_centre = mean(shear_limits(:,2)) ;


%% Full state set
[xx, yy, dd] = meshgrid(1:grid_sizex, 1:grid_sizey, 1:n_headings);
S = [xx(:), yy(:), dd(:)];



%% Action set
global a_set
a_set = {'fl', 'ff', 'fr'}; % forward-left, forward-forward, forward-right


%% Wind shear model
wind = @(S, ws, wl) [(S(:,2)-shear_centre)*ws.*...
    ((S(:,1)>wl(1))&(S(:,1)<wl(2))&(S(:,2)>wl(3))&(S(:,2)<wl(4))),...
    zeros(size(S, 1), 1)];

% Show wind plot
W = wind([XH(:), YH(:)], shear_strength, square2hex(shear_limits));
E_therm_plot = base_mass*g*delta_t*...
    source_energy(square2hex(source_locations), source_strength, ...
    source_radius, [XH(:), YH(:)]);


%% Reward from wind shear
rt3o4ws = sqrt(3)/4*grid_spacing*shear_strength*[0; 1; -1; 0; 1; -1];
halfm = 0.5*base_mass;
E_wind = 0.5*base_mass*(rt3o4ws.^2 + 2*base_speed*rt3o4ws);
E_wind = repmat(E_wind', [size(W, 1), 1]);
E_wind = E_wind(:).*repmat(logical(W(:,1)), [n_headings, 1]);

% Reward from thermal soaring
E_therm = repmat(E_therm_plot, [n_headings,1]);


%% Reward from drag
E_drag = -grid_spacing/L_on_D*base_mass*g;
E_turn = E_drag/3;
E_edge = -max(E_wind)*2;

% Compute rewards
    r_edge = -(abs(E_edge^1))*((any(origSprime < 1) || (origSprime(1) > grid_sizex) || ...
        (origSprime(2) > grid_sizey)) + (any(origSi < 1) || ...
        any(origSi(:,1) > grid_sizex) || any(origSi(:,2) > grid_sizey) ) ) ;
    
    r_move = 2*E_drag + (a~=2)*E_turn;
    r_wind = sum(E_wind([si_n; s_primen]));
    r_therm = sum(E_therm([si_n; s_primen]));
    
    rew = r_edge + r_move + r_wind + r_therm ;
    


end

