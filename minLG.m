function param = minLG(x,data,param,last_error,iter)

    if length(param) == 3
        A = param(1);
        s = param(2);
        snr = param(3);
        Sigma = zeros(length(x));
        for i = 1:length(data)
            for j = 1:length(data)
               Sigma(i,j) = kernel(x(:,i),x(:,j),[param(1:end-1),snr]);
            end
        end
        iSigma = Sigma^(-1);

        DKDA = zeros(length(x));
        DKDS = DKDA;
        DKDSNR = DKDA;

        for i = 1:length(data)
            for j = 1:length(data)
               DKDA(i,j) = DkernelDA(x(:,i),x(:,j),param(1:end-1),snr);
               DKDS(i,j) = DkernelDS(x(:,i),x(:,j),param(1:end-1),snr);
               DKDSNR(i,j) = DkernelDSNR(x(:,i),x(:,j),param(1:end-1),snr);
            end
        end

        dpdA = 1/2* (data*iSigma*DKDA*iSigma*data' - trace(iSigma*DKDA));
        dpdS = 1/2* (data*iSigma*DKDS*iSigma*data' - trace(iSigma*DKDS));
        dpdSNR = 1/2* (data*iSigma*DKDSNR*iSigma*data' - trace(iSigma*DKDSNR));

        delta = [dpdA  dpdS  dpdSNR];

        %line tracing
        min_error = 1000;
        min_alpha = 0;
        alpha = 0;
        dparam = param + delta*alpha;
        error = lg(x,data,dparam);
        if error < min_error
            min_error = error;
            min_alpha = alpha;
        end

        alpha = 1/4;
        while(alpha > 1e-5)
            dparam = param + delta*alpha;
            error = lg(x,data,dparam);
            if isreal(error)
                if error < min_error
                    min_error = error;
                    min_alpha = alpha;
                end
            end
            alpha = alpha/2;
        end

        %stop condition
        
        if (last_error - min_error < 1e-6)
            return;
        end
        if iter > 15
            return;
        end
        iter = iter +1;
        param = minLG(x,data,param+delta*min_alpha,min_error,iter);
    else %length 2
        
        s = param(1);
        snr = param(2);
        Sigma = zeros(length(x));
        for i = 1:length(data)
            for j = 1:length(data)
               Sigma(i,j) = kernel(x(:,i),x(:,j),[param(1:end-1),snr]);
            end
        end
        iSigma = Sigma^(-1);

        DKDA = zeros(length(x));
        DKDS = DKDA;
        DKDSNR = DKDA;

        for i = 1:length(data)
            for j = 1:length(data)
               DKDS(i,j) = DkernelDS(x(:,i),x(:,j),s,snr);
               DKDSNR(i,j) = DkernelDSNR(x(:,i),x(:,j),s,snr);
            end
        end

        dpdS = 1/2* (data*iSigma*DKDS*iSigma*data' - trace(iSigma*DKDS));
        dpdSNR = 1/2* (data*iSigma*DKDSNR*iSigma*data' - trace(iSigma*DKDSNR));

        delta = [dpdS  dpdSNR];

        %line tracing
        min_error = 1000;
        min_alpha = 0;
        alpha = 0;
        dparam = param + delta*alpha;
        error = lg(x,data,dparam);
        if error < min_error
            min_error = error;
            min_alpha = alpha;
        end

        alpha = 1/4;
        while(alpha > 1e-5)
            dparam = param + delta*alpha;
            error = lg(x,data,dparam);
            if isreal(error)
                if error < min_error
                    min_error = error;
                    min_alpha = alpha;
                end
            end
            alpha = alpha/2;
        end

        %stop condition
        
        if (last_error - min_error < 1e-5)
            return;
        end
        if iter > 15
            return;
        end
        iter = iter +1;
        param = minLG(x,data,param+delta*min_alpha,min_error,iter);
    end


end