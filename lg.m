function error = lg(x,data,param)
    
    A = param(1);
    s = param(2);
    snr = param(3);
    Sigma = zeros(length(x));
    for i = 1:length(data)
        for j = 1:length(data)
           Sigma(i,j) = kernel(x(:,i),x(:,j),param(1:end-1),snr);
        end
    end
    
    error = 0.5*(data*(Sigma^-1)*data' + log(det(Sigma)));





end