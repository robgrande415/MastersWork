function s_new = gridworld_trans(s_old,action,params)

N_grid = params.N_grid;
noise = params.noise;

r = sample_discrete([noise 1-noise]);


if (r == 1) % Step in an arbitrary direction
    
    dir = sample_discrete(0.2.*ones(5,1));

else % Step in the intended direction
   
    dir = action;
    
end
             % Get The Next State
            
            s_new = zeros(2,1);
            
            switch dir
               
                case 1 % Null
                    s_new(1) = s_old(1); 
                    s_new(2) = s_old(2);
                    
                case 2 % Right
                    s_new(1) = s_old(1) + 1; 
                    s_new(2) = s_old(2);
                    
                case 3 % Up
                    s_new(1) = s_old(1); 
                    s_new(2) = s_old(2) + 1;
                    
                case 4 % Left
                    s_new(1) = s_old(1) - 1; 
                    s_new(2) = s_old(2);
                    
                case 5 % Down
                    s_new(1) = s_old(1); 
                    s_new(2) = s_old(2) - 1;
                                       
                    
            end
            
            
         
         % Saturate the states if on boundaries
            
            s_new(1) = max([1,min([N_grid,s_new(1)])]);
            s_new(2) = max([1,min([N_grid,s_new(2)])]);
                      

end