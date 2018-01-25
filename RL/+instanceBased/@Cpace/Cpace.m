classdef Cpace < handle
    %CPACE Summary of this class goes here
    %   Detailed explanation goes here
    
    properties (Access = public)
        n_Acts = 1;
        corpus;
        L = 1;
        e_tol = 0.1;
        maxHistory = 10000;
        curSize = 0;
        k =1;
        rmax =0;
        gamma = .9;
        dim =1;
        neighs;
<<<<<<< .mine
        neighs_dist;
=======
        cachedDists;
>>>>>>> .r8040
        lastVI = 0;
        lastUpdate = 0;
    end
    
    methods
        function obj = Cpace(na,el,mh,kk, rm, g,d, et)
            obj.n_Acts = na;
            obj.L = el; 
            obj.k = kk;
            obj.e_tol = et; %rm/(1-g)/el;
            obj.maxHistory = mh;
            obj.corpus = struct('s',zeros(d, obj.maxHistory), 'a',zeros(obj.maxHistory,1), 'r',zeros(obj.maxHistory,1), 'T', zeros(obj.maxHistory,1), 'Q',zeros(obj.maxHistory,1), 'hasNext',zeros(obj.maxHistory,1));
            obj.curSize  =0;
            obj.rmax = rm;
            obj.gamma = g;
            obj.dim = d;
            obj.neighs = ones(kk,mh,na) * -1;
<<<<<<< .mine
            obj.neighs_dist = ones(kk,mh,na) * -1;
=======
            obj.cachedDists = ones(kk,mh,na) * -1;
>>>>>>> .r8040
        end
        
        function [indexes,dists] = findKnn(obj,state, act)
            dists = ones(obj.k,1) * -1;
            indexes =ones(obj.k,1) * -1;
            
            
            ind = (act==obj.corpus.a);
            obj.corpus.s(:,ind)
            temp = bsxfun(@minus,obj.corpus.s(:,ind),state);
            %2-norm
            %val = sqrt(sum(temp.^2,1));
            %1-norm
            val = sum(abs(temp),1);
            
            [dd,indind] = min(val);
            temp = min([obj.k,numel(dd)]);
            dists(1:temp) = dd(1:temp);
            indexes(1:temp) = indind(1:temp);
            dists_new = dists
            %old code
            
            for i = 1:obj.curSize
                if( (~(act==obj.corpus.a(i))) || (obj.corpus.hasNext(i) ==0)|| (obj.corpus.T(i) > 1)) %1 is goal state, 2 means end of episode
                    continue;
                end
                if(max(dists) > norm(state-obj.corpus.s(:,i),1) || indexes(obj.k) == -1) %max(abs(state-obj.corpus.s(:,i))) || indexes(obj.k) == -1) 
                    avail = find(indexes==-1);
                    if(~isempty(avail))
                        indexes(avail(1)) = i;
                        dists(avail(1)) = norm(state-obj.corpus.s(:,i),1); % max(abs(state-obj.corpus.s(:,i)));
                    else
                        avail = find(dists == max(dists));
                        indexes(avail(1)) = i;
                        dists(avail(1)) =norm(state-obj.corpus.s(:,i),1); % max(abs(state-obj.corpus.s(:,i)));
                    end
                end
            end
            dists_old = dists
            
        end
        
        function q = predictQ(obj,state,act,useCache)
            
            if(nargin < 4)
                useCache = 0;
            end
            
            %old code
            %{
            if(useCache > 0)
                indexes = obj.neighs(:,useCache,act);
                dists = ones(obj.k,1) * -1;
                for j=1:obj.k
                    if(indexes(j,1) == -1)
                        dists(j,1) = -1;
                    else
                        dists(j,1) = norm(obj.corpus.s(:,useCache) - obj.corpus.s(:,indexes(j,1)),1); %max(abs(obj.corpus.s(:,useCache) - obj.corpus.s(:,indexes(j,1))));
                    end
                end
            else
                [indexes,dists] = obj.findKnn(state,act);
            end
            old_dist = dists;
            %}
            
            
            %newest code
            if(useCache > 0)
                indexes = obj.neighs(:,useCache,act);
                %dists = ones(obj.k,1) * -1;
                %ind = indexes > 0;
                
                dists = obj.neighs_dist(:,useCache,act);
                
                %if isempty(ind) == 0
                %    temp = bsxfun(@minus,obj.corpus.s(:,indexes(ind)),obj.corpus.s(:,useCache));
                    %2-norm
                    %val = sqrt(sum(temp.^2,1));
                    %1-norm
                %    val = sum(abs(temp),1);
                %    dists(ind) = val;
                %end
                
            else
                [indexes,dists] = obj.findKnn(state,act);
                ind = find(indexes > 0);
            end            
            %dists_new = dists;
            
            
            %new code
            %{
            if(useCache > 0)
                indexes = obj.neighs(:,useCache,act);
                dists = obj.cachedDists(:,useCache,act);
                %dists = ones(obj.k,1) * -1;
                %ind = find(indexes > 0);
                %if isempty(ind) == 0
                %    temp = bsxfun(@minus,obj.corpus.s(:,indexes(ind)),obj.corpus.s(:,useCache));
                    %2-norm
                    %val = sqrt(sum(temp.^2,1));
                    %1-norm
                 %   val = sum(abs(temp),1);
                 %   dists(ind) = val;
                %end
                
            else
                [indexes,dists] = obj.findKnn(state,act);
                %ind = find(indexes > 0);
            end            
            dists_old = dists;
            
            if norm(dists_old-dists_new) > 0.01
                2
            end
            %}
            
            
            %old code
            for i=1:obj.k
                if(dists(i) == -1) % ||  dists(i) > obj.e_tol)
                    q = q+ obj.rmax /(1-obj.gamma);
                else
                    ind = indexes(i);
                    temp1 = obj.rmax /(1-obj.gamma);
                    temp2 = obj.corpus.Q(ind) + obj.L * dists(i);
                    q = q + min(temp1, temp2);
                end
            end
            q = q / obj.k;
<<<<<<< .mine
=======
            %old_q = q;
>>>>>>> .r8040
            
        end
        
        
        function [v,maxa] = getMax(obj,state,useCache)
            if(nargin < 3)
                useCache = 0;
            end
            for a = 1:obj.n_Acts
                q = obj.predictQ(state,a,useCache);
                if(a==1 || q > v)
                    v = q;
                    maxa = a;
                end 
            end
        
        end
    
        function update(obj,state,act,rew,sp,term)
            %TODO: update when maxHistory reached
            [indexes,dists] = obj.findKnn(state,act);
            if(sum(dists == -1) > 0 || max(dists) > obj.e_tol)
                fprintf('hit an unknown state at %f  %f  %d \n', state(1), state(2), act);
                if(indexes(1) > 0)
                    fprintf('neighbor 1 was %f  %f at distance %f\n', obj.corpus.s(1,indexes(1)), obj.corpus.s(2,indexes(1)), dists(1));
                end
                if(indexes(2) > 0)
                    fprintf('neighbor 2 was %f  %f at distance %f\n', obj.corpus.s(1,indexes(2)), obj.corpus.s(2,indexes(2)), dists(2));
                end
                if(indexes(3) > 0)
                    fprintf('neighbor 3 was %f  %f at distance %f\n', obj.corpus.s(1,indexes(3)), obj.corpus.s(2,indexes(3)),dists(3));
                end
                obj.lastUpdate = 1;
                if(obj.curSize > obj.maxHistory)
                    fprintf('corpus full\n');
                    return
                end
                obj.curSize = obj.curSize+1;
                obj.corpus.s(:,obj.curSize) =state;
                obj.corpus.hasNext(obj.curSize) = 1;
                obj.corpus.a(obj.curSize) =act;
                obj.corpus.r(obj.curSize) =rew;
                if(term > 0)
                    obj.corpus.T(obj.curSize) = term;
                    if(term == 2)
                        obj.corpus.hasNext(obj.curSize) = 0;
                    end
                else
                     obj.corpus.T(obj.curSize) = 0;
                     obj.corpus.s(:,obj.curSize+1) =sp;
                end
                if(term < 2)
                    obj.addToNeigh(state,act,obj.curSize);
                end
                for a=1:obj.n_Acts
<<<<<<< .mine
                    [indexes,dd] = obj.findKnn(state,a);
=======
                    [indexes,dists] = obj.findKnn(state,a);
>>>>>>> .r8040
                    obj.neighs(:,obj.curSize,a) = indexes;
<<<<<<< .mine
                    obj.neighs_dist(:,obj.curSize,a) = dd;
=======
                    obj.cachedDists(:,obj.curSize,a) = dists;
>>>>>>> .r8040
                end
                
                %for i=1:obj.curSize
                %    for a=1:obj.n_Acts
                %        [indexes,~] = obj.findKnn(obj.corpus.s(:,i),a);
                %        obj.neighs(:,i,a) = indexes;
                 %   end
                %end
                %fprintf('planning\n');
                obj.pbvi(0.0001);
                %fprintf('done planning\n');
            elseif(obj.corpus.hasNext(obj.curSize) == 1 && obj.corpus.T(obj.curSize) ==0)
                obj.lastUpdate =2;
                obj.curSize = obj.curSize + 1;
                obj.corpus.T(obj.curSize) = term; %if next state is terminal then record that
                obj.corpus.hasNext(obj.curSize) = 0;
                for a=1:obj.n_Acts
<<<<<<< .mine
                    [indexes,dd] = obj.findKnn(obj.corpus.s(:,obj.curSize),a);
=======
                    [indexes,dists] = obj.findKnn(obj.corpus.s(:,obj.curSize),a);
>>>>>>> .r8040
                    obj.neighs(:,obj.curSize,a) = indexes;
<<<<<<< .mine
                    obj.neighs_dist(:,obj.curSize,a) = dd;
=======
                    obj.cachedDists(:,obj.curSize,a) = dists;
>>>>>>> .r8040
                end
                obj.pbvi(0.0001);
            %else
            %    ni = obj.pbvi(0.0001);
             %   if(ni > 1)
              %      2 +2
              %  end
            end
            
        end
        
        function addToNeigh(obj,state,act,ind)
            for i=1:obj.curSize
                if(~isempty(find(obj.neighs(:,i,act)==ind, 1)))
                    continue;
                end
                dist = norm(obj.corpus.s(:,i) - state,1); %max(abs(obj.corpus.s(:,i) - state));
                dists = obj.cachedDists(:,i,act); %ones(obj.k,1) * -1;
                counted = 0;
                for kk=1:obj.k
                    if(obj.neighs(kk,i,act) > -1)
                        %dists(kk) = norm(obj.corpus.s(:,obj.neighs(kk,i,act)) -obj.corpus.s(:,i),1); %max(abs(obj.corpus.s(:,obj.neighs(kk,i,act)) -obj.corpus.s(:,i)));
                        counted = counted +1;
                    end
                end
                if(counted < obj.k)  %open slot
                    obj.neighs(counted+1,i,act) = ind;
<<<<<<< .mine
                    obj.neighs_dist(counted+1,i,act) = dist;
=======
                    obj.cachedDists(counted+1,i,act) = dist;
>>>>>>> .r8040
                elseif(max(dists) > dist)
                    maxinds = find(dists == max(dists));
                    maxind = maxinds(1);
                    obj.neighs(maxind,i,act) = ind; 
<<<<<<< .mine
                    obj.neighs_dist(maxind,i,act) = dist;
=======
                    obj.cachedDists(maxind,i,act) = dist; 
>>>>>>> .r8040
                end
                
            end

        end
        
        function [numIt] = pbvi(obj, tol)
            obj.lastVI = obj.curSize;
            delta = tol + 1;
            numIt = 1;
            while(delta > tol)
                delta = 0;
                for i=1:obj.curSize
                    if(obj.corpus.T(i) == 1)
                        delta = max(abs(obj.corpus.Q(i) - obj.corpus.r(i)), delta);
                        obj.corpus.Q(i) = obj.corpus.r(i);
                        continue;
                    elseif(obj.corpus.hasNext(i) ==0)
                        continue;
                    end
                    if(i < obj.curSize) %AND the next guy has neighbors
                        cache = i+1;
                    else
                        cache = 0;
                    end
                    [val,~] = obj.getMax(obj.corpus.s(:,i+1),cache);
                    oldQ = obj.corpus.Q(i);
                    obj.corpus.Q(i) = obj.corpus.r(i) + obj.gamma * val;
                    delta = max(abs(oldQ - obj.corpus.Q(i)), delta);
                end
                numIt = numIt+1;
            end
        end
        
        
    end
end
