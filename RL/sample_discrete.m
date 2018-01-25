function y = sample_discrete(p)

%sample a multivariable distribution
% with probability vector p

n = length(p);

j = 1;
can = zeros(1,n+1);
ind = zeros(1,n+1);
can(1) = 0; 


for i = 1:n
    if(p(i) > 0)
        can(j+1) = p(i) + can(j);
        ind(j) = i;
        j = j+1;
        
    end
end

w = rand();
%disp(w);

c = 1;

for i = 2:n+1
    if(w<=can(i))
        c = min(c,can(i));
    end
end

for j = 2:n+1
    if(can(j) == c)
        
        break;
    end
end


 l = ind(j-1);
 
y = l;

end