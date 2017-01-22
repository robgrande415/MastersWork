function [x,y,f,act_model] = get_data(inFileName)
    act_model = [];
    f = [];
switch inFileName
    case 'Bee_waggle1'
        load('../data/bee_waggle/sequence1/btf/ximage.btf');
        load('../data/bee_waggle/sequence1/btf/yimage.btf');
        load('../data/bee_waggle/sequence1/btf/timage.btf');
        timage = wrap(timage);
        %timage = smooth(timage);
        
        %x = [cos(timage)'; sin(timage)'];
        x = timage';
        %get more data
        dx = diff(x);
        xtemp = [];
        for i = 1:length(x)-1
            xtemp = [xtemp, x(:,i),x(:,i)+dx(:,i)/2];
        end
        xtemp = [xtemp, x(:,end)];
        x = xtemp;
        y = diff(x)*2;
        x = x(2:end);
        %y = [timage'];
    case 'Walk'
        D = amc_to_matrix('../data/CMU/walk1.amc');
        locations = [1 7 10 13 16 19 22 25 27 30 31 32 34 35 37 39 42 43 44 46 47 49 52 53 55 56 59 60 62];
        ind = locations([26,27,28]);
        x = D(:,ind)';
        y = diff(x')';
        y = y(1,:);
        x = x(:,1:end-1);
        y = smooth(y)';
        
        %Changepoint backwards
        x2 = fliplr(D(:,ind)');
        y2 = diff(x2')';
        y2 = y2(1,:);
        x2 = x2(:,1:end-1);
        y2 = smooth(y2)';
        x = [x,x2];
        y = [y,y2];
        
    case 'Variable Walk'
        D = amc_to_matrix('../data/CMU/variousWalk1.amc');
        locations = [1 7 10 13 16 19 22 25 27 30 31 32 34 35 37 39 42 43 44 46 47 49 52 53 55 56 59 60 62];
        ind = locations([22,23,24,26,27,28]);
        x = D(:,ind(4):ind(end))';
        y = diff(x')';
        y = y(1,:);
        x = x(:,1:end-1);
        y = smooth(y)';
        
    case 'Walk Turn'
        D = amc_to_matrix('../data/CMU/walkTurn1.amc');
        locations = [1 7 10 13 16 19 22 25 27 30 31 32 34 35 37 39 42 43 44 46 47 49 52 53 55 56 59 60 62];
        ind = locations([3,4,22,23,24,26,27,28]);
        x = D(:,ind(3):ind(5))';
        y = D(:,ind(3):ind(5))';
        y = y(2,:);
        y = diff(y')';
        x = x(:,1:end-1);
        y = smooth(y)';
        
    case 'Run Leap'
        D = amc_to_matrix('../data/CMU/runleap1.amc');
        locations = [1 7 10 13 16 19 22 25 27 30 31 32 34 35 37 39 42 43 44 46 47 49 52 53 55 56 59 60 62];
        ind = locations([9,10,11,22,23,24]);
        x = D(:,ind)';
        y = D(:,ind)';
        y = y(2,:);
        y = diff(y')';
        x = x(:,1:end-1);
        y = smooth(y)';
        %repeat
        x = repmat(x,1,2);
        y = repmat(y,1,2);
        
    case 'Walk Run'
        
        %walking
        D = amc_to_matrix('../data/CMU/walk1.amc');
        locations = [1 7 10 13 16 19 22 25 27 30 31 32 34 35 37 39 42 43 44 46 47 49 52 53 55 56 59 60 62];
        ind = locations([3,4,22,23,24,26,27,28]);
        x = D(:,ind(3):ind(5))';
        y = D(:,ind(3):ind(5))';
        y = y(2,:);
        y = diff(y')';
        x = x(:,1:end-1);
        y = smooth(y)';
        %running
        D = amc_to_matrix('../data/CMU/run1.amc');
        locations = [1 7 10 13 16 19 22 25 27 30 31 32 34 35 37 39 42 43 44 46 47 49 52 53 55 56 59 60 62];
        ind = locations([3,4,22,23,24,26,27,28]);
        x2 = D(:,ind(3):ind(5))';
        y2 = D(:,ind(3):ind(5))';
        y2 = y2(2,:);
        y2 = diff(y2')';
        x2 = x2(:,1:end-1);
        y2 = smooth(y2)';
        
        x = [x,x2];
        y = [y,y2];
        
        
    case 'Variety3'
        
        %walking
        D = amc_to_matrix('../data/CMU/walk1.amc');
        locations = [1 7 10 13 16 19 22 25 27 30 31 32 34 35 37 39 42 43 44 46 47 49 52 53 55 56 59 60 62];
        ind = locations([3,4,22,23,24,26,27,28]);
        x = D(:,ind)';
        y = D(:,ind)';
        y = y(2,:);
        y = diff(y')';
        x = x(:,1:end-1);
        y = smooth(y)';
        
        %foward jump
        D = amc_to_matrix('../data/CMU/forward_jump1.amc');
        locations = [1 7 10 13 16 19 22 25 27 30 31 32 34 35 37 39 42 43 44 46 47 49 52 53 55 56 59 60 62];
        ind = locations([3,4,22,23,24,26,27,28]);
        x2 = D(:,ind)';
        y2 = D(:,ind)';
        y2 = y2(2,:);
        y2 = diff(y2')';
        x2 = x2(:,1:end-1);
        y2 = smooth(y2)';
        x2 = x2(:,100:end);
        y2 = y2(:,100:end);
        %high jump
        D = amc_to_matrix('../data/CMU/high_jump1.amc');
        locations = [1 7 10 13 16 19 22 25 27 30 31 32 34 35 37 39 42 43 44 46 47 49 52 53 55 56 59 60 62];
        ind = locations([3,4,22,23,24,26,27,28]);
        x3 = D(:,ind)';
        y3 = D(:,ind)';
        y3 = y3(2,:);
        y3 = diff(y3')';
        x3 = x3(:,1:end-1);
        y3 = smooth(y3)';
        
        x = [x,x2,x];
        y = [y,y2,y];
        
        case 'RobotDance1'
        
            %walking
            time_index = 1:1000;
            x_ind = 2:4;
            y_ind = 5;
            robot1 = load('../data/RobotDance/GP05_0_data_cleaned.log');
            robot2 = load('../data/RobotDance/GP07_0_data_cleaned.log');
            x1 = [robot1(:,x_ind)'-robot2(:,x_ind)'];
            y1 = robot2(:,y_ind)';
            x1 = x1(:,time_index);
            y1 = y1(:,time_index);
            %walking
            robot1 = load('../data/RobotDance/GP05_1_data_cleaned.log');
            robot2 = load('../data/RobotDance/GP07_1_data_cleaned.log');
            x2 = robot1(:,x_ind)'-robot2(:,x_ind)';
            x2 = x2(:,time_index);
            y2 = robot2(:,y_ind)';
            y2 = y2(:,time_index);
            %walking
            robot1 = load('../data/RobotDance/GP05_6_data_cleaned.log');
            robot2 = load('../data/RobotDance/GP07_6_data_cleaned.log');
            x3 = robot1(:,x_ind)'-robot2(:,x_ind)';
            x3 = x3(:,time_index);
            y3 = robot2(:,y_ind)';
            y3 = y3(:,time_index);
            %walking
            robot1 = load('../data/RobotDance/GP05_4_data_cleaned.log');
            robot2 = load('../data/RobotDance/GP07_4_data_cleaned.log');
            x4 = robot1(:,x_ind)'-robot2(:,x_ind)';
            x4 = x4(:,time_index);
            y4 = robot2(:,y_ind)';
            y4 = y4(:,time_index);
            
            %x = [x1,x2,x1,x3,x1,x4,x3,x2,x1];
            %y = [y1,y2,y1,y3,y1,y4,y3,y2,y1];
            
            %
            x = [x1,x2,x1,x3,x1,x4,x2,x3,x1];
            y = [y1,y2,y1,y3,y1,y4,y2,y3,y1];
            load good_data_robots4

    case 'Airplane'
        inFileName1 = '../data/airplane/ta20100721/ta20100721_1';
        %25% off
        inFileName2 = '../data/airplane/ta20091002/ta20091002_flight3_25wingoff_1';
        %50% off
        inFileName3 = '../data/airplane/ta20091002/ta20091002_flight3_50wingoff_1';
        load(inFileName3);
        time1=time1-time1(1);
        lim=0.7;

        %data legend
        %stop=5;
        stop=5000;
        shifter=1;
        cmd=time1*0+200;

        phi = innerloop_e0(1:end-stop);
        alpha = innerloop_e1(1:end-stop);
        beta = innerloop_e2(1:end-stop);
        x_coord = navstate_x0(shifter:end-stop);
        y_ccord = navstate_x1(shifter:end-stop);
        altitude = -navstate_x2(shifter:end-stop);
        
        
        rudder = actuatorIntWork_delm0(shifter:end-stop);
        elevator = actuatorIntWork_delm1(shifter:end-stop)*-1;
        aileron = actuatorIntWork_delm2(shifter:end-stop)*-1;
        throttle = (actuatorIntWork_delf0(shifter:end-stop)*100+100)/2; 
        
        %outputs of velocity
        z_vel = diff(altitude)*50;
        z_vel = [z_vel', z_vel(end)];
        x = [(rudder'-mean(rudder))/std(rudder);(elevator'-mean(elevator))/std(elevator);...
                (aileron'-mean(aileron))/std(aileron);(throttle'-mean(throttle))/std(throttle)];
        y = smooth(z_vel,5)';
    case 'Diffusion'
        change_ind = [0 75 125 225 350 450 570 630 700 800]*3;
        stop_ind = 1000*3;
        A1 = [0.1 0.3 0.5 0.7 0.9];
        L1 = 0.9;
        cluster_num = [1 2 1 3 4 2 4 3 1 5];
        x1 = zeros(1,stop_ind);
        y1 = x1;
        v = 0.2;
        wn = 0.01;
        act_model = x1;
        act_model(1) = 1;
        for ii = 2:stop_ind
            curr_num = cluster_num(max(find(ii>change_ind)));
            act_model(ii) = curr_num;
            A = A1(curr_num);
            x1(ii) = x1(ii-1)*L1 + A + randn*v;
            y1(ii) = x1(ii) +randn*wn;
        end
       
        %get data for CPD
        stop = 3000;
        x = x1(1:stop-1)*0;
        y = x1(2:stop);
        %y = diff(x1);
        
        case 'Diffusion2'
            change_ind = [0 100 200];
            stop_ind = 200;
            A1 = [0 0.2 0];
            L1 = 0.8;
            cluster_num = [1 2 1 3 4 2 4 3 1 5];
            x1 = zeros(1,stop_ind);
            y1 = x1;
            v = 0.02;
            wn = 0.1;
            act_model = x1;
            act_model(1) = 1;
            for ii = 2:stop_ind
                curr_num = cluster_num(max(find(ii>change_ind)));
                act_model(ii) = curr_num;
                A = A1(curr_num);
                x1(ii) = x1(ii-1)*L1 + A + randn*v;
                y1(ii) = x1(ii);
            end

            %get data for CPD
            stop = stop_ind;
            x = x1(1:stop)*0;
            f = x1;
            y = y1(1:stop)+randn(size(y1))*wn;
            %y = diff(x1);
            
    case 'Parabola'
        change_ind = [0 75+round(rand*50) 175+round(50*rand)];
        change_ind = [0 100 200];
        stop_ind = 400;
        cluster_id = [1 2 1];
        x_temp = rand(1,stop_ind)*2-1;
        y1 = x_temp.^2-1/2;
        y2 = 1/2*x_temp.^2;
        x = zeros(1,stop_ind);
        y = x;
        wn = 0.15;
        for ii = 1:stop_ind
            rr = randi(stop_ind);
            x(ii) = x_temp(rr);
            curr_num = cluster_id(max(find(ii>change_ind)));
            act_model(ii) = curr_num;
            if curr_num ==1
                y(ii) = y1(rr);
            else
                y(ii) = y2(rr);
            end
        end
        f = y;
        y = y + randn(size(y))*wn;
        
    case 'Parabola2'
        change_ind = [0 75+round(rand*50) 175+round(50*rand)];
        change_ind = [0 100 200];
        stop_ind = 400;
        cluster_id = [1 2 1];
        x_temp = rand(1,stop_ind)*2-1;
        y1 = x_temp.^2;
        y2 = -x_temp.^2+2;
        x = zeros(1,stop_ind);
        y = x;
        wn = 0.15;
        for ii = 1:stop_ind
            rr = randi(stop_ind);
            x(ii) = x_temp(rr);
            curr_num = cluster_id(max(find(ii>change_ind)));
            act_model(ii) = curr_num;
            if curr_num ==1
                y(ii) = y1(rr);
            else
                y(ii) = y2(rr);
            end
        end
        f = y;
        y = y + randn(size(y))*wn;

end
end

function x = wrap(x)
    for i = 2:length(x)
        if (x(i) - x(i-1)) > pi
            x(i:end) = x(i:end)-pi;
        elseif (x(i) - x(i-1)) < -pi
            x(i:end) = x(i:end)+pi;
        end
    end

end