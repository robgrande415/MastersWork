switch filename
    case 'Diffusion2'
        figure(2)
        hold on
        if BOCPD_flag
            plot(BOCPD{1}.mean,'r')
        end
        if NBHT_flag
            plot(NBHT{1}.mean,'g')
        end
        if DPGP_flag
            plot(DPGP{1}.mean,'c')
        end
        if MCMC_flag
            plot(MCMC{1}.mean,'k')
        end

        
        %{
        figure(3)
        hold on
        plot(BOCPD{1}.mean-f,'r')
        plot(NBHT.mean-f,'g')
        plot(DPGP{1}.mean-f,'c')
        %}
        if BOCPD_flag
            mean = mean(abs(BOCPD{1}.mean-f))
        end
        if NBHT_flag
            mae_NBHT = mean(abs(NBHT{1}.mean-f))
        end
        if DPGP_flag
            mae_DPGP = mean(abs(DPGP{1}.mean-f))
        end
        if MCMC_flag
            mae_MCMC = mean(abs(MCMC{1}.mean-f))
        end
        
        
    case {'Parabola','Parabola2'}
        figure(2)
        subplot(1,3,1)
        hold on
        if BOCPD_flag
        plot(x(1:100),BOCPD{1}.mean(1:100),'rx')
        end
        if FORGET_flag
        plot(x(1:100),FORGET{1}.mean(1:100),'rx')
        end
        if NBHT_flag
        plot(x(1:100),NBHT{1}.mean(1:100),'gx')
        end
        if DPGP_flag
        plot(x(1:100),DPGP{1}.mean(1:100),'cx')
        end
        if MCMC_flag
        plot(x(1:100),MCMC{1}.mean(1:100),'kx')
        end
        x_test = linspace(-1,1,200);
        plot(x_test,x_test.^2-1/2,'x')
        subplot(1,3,2)
        hold on
        if BOCPD_flag
        plot(x(100:200),BOCPD{1}.mean(100:200),'rx')
        end
        if FORGET_flag
        plot(x(100:200),FORGET{1}.mean(100:200),'rx')
        end
        if NBHT_flag
        plot(x(100:200),NBHT{1}.mean(100:200),'gx')
        end
        if DPGP_flag
        plot(x(100:200),DPGP{1}.mean(100:200),'cx')
        end
        if MCMC_flag
        plot(x(100:200),MCMC{1}.mean(100:200),'kx')
        end
        x_test = linspace(-1,1,200);
        plot(x_test,1/2*x_test.^2,'x')
        subplot(1,3,3)
        hold on
        if BOCPD_flag
        plot(x(200:400),BOCPD{1}.mean(200:400),'rx')
        end
        if FORGET_flag
        plot(x(200:400),FORGET{1}.mean(200:400),'rx')
        end
        if NBHT_flag
        plot(x(200:400),NBHT{1}.mean(200:400),'gx')
        end
        if DPGP_flag
        plot(x(200:400),DPGP{1}.mean(200:400),'cx')
        end
        if MCMC_flag
        plot(x(200:400),MCMC{1}.mean(200:400),'kx')
        end
        x_test = linspace(-1,1,200);
        plot(x_test,x_test.^2-1/2,'x')
        %figure(3)
        %hold on

        %plot(BOCPD{1}.mean-f,'r')
        %plot(NBHT{1}.mean-f,'g')
        if BOCPD_flag
           BOCPD_stats.mean =0;
           BOCPD_stats.time =0;
            for i = 1:length(BOCPD)
                BOCPD_stats.mean = BOCPD_stats.mean+mean(abs(BOCPD{i}.mean-BOCPD{i}.f))/length(BOCPD);
                bocpd_mean(i) = mean(abs(BOCPD{i}.mean-BOCPD{i}.f));
                BOCPD_stats.time = BOCPD_stats.time+BOCPD{i}.time/length(BOCPD);
            end
            BOCPD_stats.mean;
            BOCPD_stats.time;
            BOCPD_stats
        end
        if FORGET_flag
           FORGET_stats.mean =0;
           FORGET_stats.time =0;
            for i = 1:length(FORGET)
                FORGET_stats.mean(i) = mean(abs(FORGET{i}.mean-FORGET{i}.f));
                FORGET_stats.time(i) = FORGET{i}.time;
            end
            mean(FORGET_stats.mean)
            std(FORGET_stats.mean)
            mean(FORGET_stats.time)
            
            FORGET_stats.mean;
            FORGET_stats.time;
            FORGET_stats;
        end
        if NBHT_flag
            NBHT_stats.mean =0;
            NBHT_stats.time=0;
            NBHT_stats.clusters=0;
            for i = 1:length(NBHT)
                NBHT_stats.mean = NBHT_stats.mean+mean(abs(NBHT{i}.mean-NBHT{i}.f))/length(NBHT);
                nbht_mean(i) = mean(abs(NBHT{i}.mean-NBHT{i}.f));
                NBHT_stats.time = NBHT_stats.time+NBHT{i}.time/length(NBHT);
                if (NBHT{i}.est_model(end) == 1)
                    NBHT_stats.clusters = NBHT_stats.clusters +2/length(NBHT);
                    NBHT_clusters(i) = 2;
                else
                    NBHT_stats.clusters = NBHT_stats.clusters +3/length(NBHT);
                    NBHT_clusters(i) = 3;
                end
            end
            NBHT_stats.mean;
            NBHT_stats.time;
            NBHT_stats.clusters;
            NBHT_stats
        end
        if DPGP_flag
            DPGP_stats.mean =0;
             DPGP_stats.time =0;
             DPGP_stats.clusters =0;
            for i = 1:length(DPGP)
                dpgp_mean(i) = mean(abs(DPGP{i}.mean-DPGP{i}.f));
                dpgp_clusters(i) = length(unique(DPGP{i}.cluster_id));
                DPGP_stats.mean = DPGP_stats.mean+mean(abs(DPGP{i}.mean-DPGP{i}.f))/length(DPGP);
                DPGP_stats.time = DPGP_stats.time+DPGP{i}.time/length(DPGP);
                DPGP_stats.clusters = DPGP_stats.clusters + length(unique(DPGP{i}.cluster_id))/length(DPGP);
            end
            DPGP_stats.mean;
            DPGP_stats.time;
            DPGP_stats.clusters;
            DPGP_stats
        end
        if MCMC_flag
            MCMC_stats.mean =0;
            MCMC_stats.time =0;
            MCMC_stats.clusters =0;
            for i = 1:length(MCMC)
                mcmc_mean(i) = mean(abs(MCMC{i}.mean-BOCPD{i}.f));
                mcmc_clusters(i) = length(unique(MCMC{i}.cluster_id));
                MCMC_stats.mean = MCMC_stats.mean+mean(abs(MCMC{i}.mean-MCMC{i}.f))/length(MCMC);
                MCMC_stats.time = MCMC_stats.time+MCMC{i}.time/length(MCMC);
                MCMC_stats.clusters = MCMC_stats.clusters + length(unique(MCMC{i}.cluster_id))/length(MCMC);
            end
            MCMC_stats.mean;
            MCMC_stats.time;
            MCMC_stats.clusters;
            MCMC_stats
        end



    case 'RobotDance1'
        if NBHT_flag
           NBHT_stats.mean = mean(abs(NBHT{1}.mean-y));
           NBHT_stats.var = NBHT{1}.var;
           for i =1:length(y)
            nllh(i) = -log(normpdf(y(i),NBHT{1}.mean(i),(NBHT_stats.var(i)+0.02)^0.5));
           end
            NBHT_stats.NLLH = mean(nllh);
            NBHT_stats
        end
        if BOCPD_flag
            last_indx = 15;
           BOCPD_stats.mean = mean(abs(BOCPD{1}.mean(1:last_indx)-y(1:last_indx)));
           BOCPD_stats.var = BOCPD{1}.stdev(1:last_indx);
           for i =1:last_indx
            nllh(i) = -log(normpdf(y(i),BOCPD{1}.mean(i),(BOCPD_stats.var(i)+0.02)^0.5));
           end
            BOCPD_stats.NLLH = mean(nllh);
            BOCPD_stats
        end
        
        





end
