classdef onlineGP_MA < handle  
    %ONLINEGP_MA A multi-action GP class
    %   multiple action version of the GP class
    
    properties (Access = public)
        n_Acts = 1;
        agp;
    end
    
    methods
        
        function obj = onlineGP_MA(na,sigma,noise,ncent,tol,A)
            obj.n_Acts = na;
            obj.agp = onlineGP.onlineGP_CopyAndReinit.empty(na,0);
            for i=1:na
            % now call superclass constructor 
                obj.agp(i) = onlineGP.onlineGP_CopyAndReinit(sigma,noise,ncent,tol,A);
            end
        end
        
     function [pred_mean,pred_var] = predict(obj,test_data,action,covar_type)
         if(nargin < 4)
            covar_type = 'full';
         end
        [pred_mean, pred_var] = obj.agp(action).predict(test_data,covar_type);
     end
        
     function update(obj,x,action,y)
         %if this is the first datapoint for that gp, then process instead
         %of update
        
         if(obj.agp(action).get('size') == 0)
             disp('first time for action');
             obj.agp(action).process(x,y);
             
         else
             obj.agp(action).update(x,y);
         end
     end
      
     function x=getOnlyAct(obj,f,action)
        x = obj.agp(action).get(f);
     end
     
        
    function x=getField(obj,f)
        for i=1:obj.n_Acts
            x(i) = obj.agp(i).get(f);
        end
    end

     
    function x=getGP(obj,a)
            x = obj.agp(a);
    end
    
    function h = visualize(obj,x,fig_struct,a)
        fig_struct = [];
        h = obj.agp(a).visualize(x,fig_struct);
    end
    
    function [Q,a_index]=getMax(obj,x)
        covar_type = 'full';
        bestAct = zeros(obj.n_Acts,1);
        bestAct(1,1) = 1;
        [Q,v] = obj.agp(1).predict(x,covar_type);
        a_index =1;
        for i=2:obj.n_Acts
            [Q2,v2] = obj.agp(i).predict(x,covar_type);
            if(Q2 > Q)
                Q = Q2;
                bestAct = bestAct * 0;
                bestAct(i,1) = 1; 
            elseif (Q2 ==Q)
                bestAct(i,1) = 1; 
            end  
        end
        inds = find(bestAct==1);
        a_index =  inds(randi(size(inds,1)));
    end
    
    %TODO: Make this take in/out arguments so we don't have to worry about
    %realloating or having a copy constructor.
    function copy(obj, oldGP,a)
        x.n_Acts = oldGP.n_Acts;
        x.agp = onlineGP.onlineGP_CopyAndReinit.empty(obj.n_Acts,0);
        if(nargin == 1)
            for i=1:obj.n_Acts
                obj.agp(i).copy(oldGP.agp(i));
            end
        else
            obj.agp(a).copy(oldGP.agp(a));
        end
    end 
    
    
    function reinitCovar(obj,a )
            if(nargin == 1)
                for i=1:obj.n_Acts
                    obj.agp(i).reinitCovar();
                end
            else
                obj.agp(a).reinitCovar();
            end
    end
        
    function replace_BV(obj,x,y,a )
            if(nargin < 4)
                for i=1:obj.n_Acts
                    obj.agp(i).replace_BV(x,y);
                end
            else
                obj.agp(a).replace_BV(x,y);
            end
    end
        
    
    
    end
    
    
end

