function params = get_params(filename,algo_type)


switch filename
    case  'RobotDance1'

        params.bandwidth = 1.5516; 
        params.tol = 1e-4; 
        params.noise = sqrt(0.02 );
        params.parameter_est = 0;
        params.max_points = 100;
        params.detect_size = 20;
        params.kl_tol = 1;
        params.bin_tol = 0.5;

    case 'Airplane'
        %airplace

        params.bandwidth = 0.3085; 
        params.tol = 1e-4; 
        params.noise = sqrt(2.04 );
        params.parameter_est = 0;
        params.max_points = 100;
        params.detect_size = 30;
        params.kl_tol = 100;
        params.bin_tol = inf;

    case 'Diffusion'

        params.bandwidth = 1; 
        params.tol = 1e-4; 
        params.noise = sqrt(0.6 );
        params.parameter_est = 0;
        params.max_points = 2;
        params.detect_size = 20;
        params.kl_tol = 0.5;
        params.bin_tol = 1;

    case 'Diffusion2'

        params.bandwidth = 1; 
        params.tol = 1e-4; 
        params.noise = sqrt(0.05 );
        params.parameter_est = 0;
        params.max_points = 1;
        params.detect_size = 10;
        params.kl_tol = 0.1;
        params.bin_tol = 1;

    case {'Parabola','Parabola2'}

        params.bandwidth = 1; 
        params.tol = 1e-4; 
        params.noise = 0.15;
        params.parameter_est = 0;
        params.max_points = 10;
        params.detect_size = 10;
        params.kl_tol = 0.5;
        params.bin_tol = 1;


end
        
switch algo_type
    case 'CRP'      
        params.detect_size = 1;
        params.kl_tol = 0.2;
end


end

