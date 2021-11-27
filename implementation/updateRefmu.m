function [occ]= updateRefmu(response_diff,frame)
   
        [occ]=varphiFunction(response_diff,frame);
end
function [occ]=varphiFunction(response_diff,frame)
%         global eta_list;
        phi=0.2; %0.3
        eta=norm(response_diff(1)+response_diff(2)+response_diff(3)+response_diff(4)+response_diff(5),2)/5e4;
%         eta_list(frame)=eta;
        if eta<phi
            occ=false;
        else
            occ=true;
        end
end