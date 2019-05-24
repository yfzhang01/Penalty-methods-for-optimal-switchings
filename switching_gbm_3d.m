function [res1,res2,res3]=switching_gbm_3d(rho,c)


% use Dirichet BC u=0 at x_max
%
tic;
dx=1/50;
sigma=0.2; mu=0.06;r=0.02; 





x_min=0;x_max=2;
n=(x_max-x_min)/dx;
x=[x_min:dx:x_max-dx]';


%BC: u=0 at x=x_max.
% u_xx central difference
e=ones(n,1); 
h=spdiags([e -2*e e],-1:1,n,n); 
 h=h/(dx^2); %h=al*h;

% u_x forward difference 
d=spdiags([-e e],[0 1],n,n);
d=d/(dx);

alpha=0.5;

x_mat=spdiags(x,0,n,n);
B1=-r*x_mat*d;
A2=-0.5*alpha^2*sigma^2*(x_mat.^2)*h;
B2=-(r+alpha*(mu-r))*x_mat*d;
A3=-0.5*sigma^2*(x_mat.^2)*h;
B3=-mu*x_mat*d;

L=[B1,sparse(n,2*n);sparse(n,n), A2+B2, sparse(n,n); sparse(n,2*n),A3+B3]; % combined generator
lambda=r*speye(3*n,3*n);
A=L+lambda;

%source
l=(-(x-0.5).*(x<=0.5)+(x-0.5).*(x>0.5).*(x<=1)+(-1)*(x-1.5).*(x>1).*(x<=1.5)+(x-1.5).*(x>1.5).*(x<=1.75));
l=repmat(l,3,1); %%u=[u0;u1];
u_old=A\l; 


% policy iteration

tol=10^(-9); 
itr_n=0;
err_n=1;
while err_n> tol

        p_old=[pi(u_old(n+1:2*n)-c-u_old(1:n))+pi(u_old(2*n+1:3*n)-c-u_old(1:n));...
            pi(u_old(1:n)-c-u_old(n+1:2*n))+pi(u_old(2*n+1:3*n)-c-u_old(n+1:2*n));...
            pi(u_old(1:n)-c-u_old(2*n+1:3*n))+pi(u_old(n+1:2*n)-c-u_old(2*n+1:3*n))];


        G_old=A*u_old-rho*p_old-l;


        P11=spdiags(-1*pi_dev(u_old(n+1:2*n)-c-u_old(1:n))-1*pi_dev(u_old(2*n+1:3*n)-c-u_old(1:n)),0,n,n);
        P12=spdiags(pi_dev(u_old(n+1:2*n)-c-u_old(1:n)),0,n,n); P13=spdiags(pi_dev(u_old(2*n+1:3*n)-c-u_old(1:n)),0,n,n);

        P22=spdiags(-1*pi_dev(u_old(1:n)-c-u_old(n+1:2*n))-1*pi_dev(u_old(2*n+1:3*n)-c-u_old(n+1:2*n)),0,n,n);
        P21=spdiags(pi_dev(u_old(1:n)-c-u_old(n+1:2*n)),0,n,n); P23=spdiags(pi_dev(u_old(2*n+1:3*n)-c-u_old(n+1:2*n)),0,n,n);

        P31=spdiags(pi_dev(u_old(1:n)-c-u_old(2*n+1:3*n)),0,n,n); P32=spdiags(pi_dev(u_old(n+1:2*n)-c-u_old(2*n+1:3*n)),0,n,n);
        P33=spdiags(-1*pi_dev(u_old(1:n)-c-u_old(2*n+1:3*n))-1*pi_dev(u_old(n+1:2*n)-c-u_old(2*n+1:3*n)),0,n,n);

        P=[P11,P12, P13;P21, P22, P23; P31, P32,P33];

        u=u_old-(A-rho*P)\G_old;
        itr_n=itr_n+1; 
        err_n=max(abs(u-u_old))/max(max(abs(u)),1) ;
        u_old=u; 
end

%% output
res1=u;
res2=itr_n;
res3=toc;

end



