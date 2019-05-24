function [res1,res2,res3]=switching_gbm_2d(rho,c)


% use Dirichet BC u=0 at x_max
%
tic;

sigma=0.2; mu=0.06;r=0.02; 


dx=1/50;


x_min=0;x_max=2;
n=(x_max-x_min)/dx;
x=[x_min:dx:x_max-dx]';


%BC: u=0 at x=x_max.
% u_xx central difference
e=ones(n,1); 
h=spdiags([e -2*e e],-1:1,n,n); 
 h=h/(dx^2); 

% u_x forward difference 
d=spdiags([-e e],[0 1],n,n);
d=d/(dx);

%
x_mat=spdiags(x,0,n,n);
B1=-r*x_mat*d;
A2=-0.5*sigma^2*(x_mat.^2)*h;
B2=-mu*x_mat*d;

L=[B1,sparse(n,n);sparse(n,n), A2+B2]; % combined generator
lambda=r*speye(2*n,2*n);
A=L+lambda;

% source

l= ((-2)*(x-1).*(x>0.75).*(x<=1)); %
%
l=[l;l]; 

% initial guess u=[u0;u1];
u_old=A\l; 


% policy iteration

tol=10^(-9); 
err=1; itr_n=0;

while err> tol 
    % linear penality
    p_old=[max(u_old(n+1:2*n)-c-u_old(1:n),0);max(u_old(1:n)-c-u_old(n+1:2*n),0)];
    
    G_old=A*u_old-rho*p_old-l;
    
    % linear penality
    P11=spdiags(-1*((u_old(n+1:2*n)-c-u_old(1:n))>0),0,n,n);
    P12=spdiags(1*((u_old(n+1:2*n)-c-u_old(1:n))>0),0,n,n);
    P21=spdiags(1*((u_old(1:n)-c-u_old(n+1:2*n))>0),0,n,n);
    P22=spdiags(-1*((u_old(1:n)-c-u_old(n+1:2*n))>0),0,n,n);
    
    P=[P11,P12;P21, P22];
    
    u=u_old-(A-rho*P)\G_old;
    itr_n=itr_n+1; err=max(abs(u-u_old))/max(max(abs(u)),1) ;
    u_old=u; 
end

% output

res1=u;  % the value function
res2=itr_n; % the total iterations
res3=toc;  % the computational time


end



