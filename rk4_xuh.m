function blah()

clc

% initial state
x0 = [1;2];

% control
u0 = 0.2;

% timestep
h0 = 24;

% number of rk4's to run in that timestep
% each rk4 step will be h0/n, so total time step is h0
n = 100;

% test this thing

% test 1
[a_x1,a_Sx1,a_Su1] = rk4_xu(x0,u0,h0,n);
[a_x1b,a_Sx1b] = finite_difference_jacob(@(x)(rk4_xu(x,u0,h0,n)), x0);
[a_x1c,a_Su1c] = finite_difference_jacob(@(u)(rk4_xu(x0,u,h0,n)), u0);

match(a_x1, a_x1b, 'test 1: rk4s')
match(a_x1, a_x1c, 'test 1: rk4s')
match(a_Sx1, a_Sx1b, 'test 1: Sx')
match(a_Su1, a_Su1c, 'test 1: Su')

% test 2
[b_x1,b_Sx1,b_Su1,b_Sh1] = rk4_xuh(x0,u0,h0,n);
[b_x1b,b_Sx1b] = finite_difference_jacob(@(x)(rk4_xuh(x,u0,h0,n)), x0);
[b_x1c,b_Su1c] = finite_difference_jacob(@(u)(rk4_xuh(x0,u,h0,n)), u0);
[b_x1d,b_Sh1d] = finite_difference_jacob(@(h)(rk4_xuh(x0,u0,h,n)), h0);

match(b_x1, b_x1b, 'test 2: rk4s')
match(b_x1, b_x1c, 'test 2: rk4s')
match(b_Sx1, b_Sx1b, 'test 2: Sx')
match(b_Su1, b_Su1c, 'test 2: Su')
match(b_Sh1, b_Sh1d, 'test 2: Sh')

end

function match(x0,x1,msg)
err = max(max(abs(x0-x1)));
if err > 1e-5
   x0
   x1
   error('%s: max error: %.3g\n', msg, err);
else
   fprintf('%s: max error: %.3g\n', msg, err);
end

end

function [f0, J] = finite_difference_jacob( fun, x0 )
% make sure x0 is a column vector
[Nx,cols] = size(x0);
if cols ~= 1
    error('x0 needs to be a column vector');
end

% make sure fun returns a column vector
f0 = fun(x0);
[Nf,cols] = size(f0);
if cols ~= 1
    error('fun needs to return a column vector');
end

% initialize empty J
J = zeros(Nf, Nx);

% perform the finite difference jacobian evaluation
%h = 1e-8;
h = 1e-12;
for k = 1:Nx
    x = x0;
    %x(k) = x(k) + h;
    x(k) = x(k) + i*h;
    f = fun(x);
    %grad = (f - f0)/h;
    grad = imag(f)/h;
    J(:,k) = grad;
end
end

function [xdot,dfdx,dfdu] = f(x,u)

p = x(1);
v = x(2);

xdot = [v; -sin(p)-0.2*v+u*u];

dfdx = [0, 1;
       -cos(p), -0.2];

dfdu = [0;
        2*u];

end

function [x1,Sx1,Su1,Sh1] = rk4_xuh(x0,u,h0,n)

h = 1/n; % n steps

nx = length(x0);
nu = length(u);

Sx0 = eye(nx);
Su0 = zeros(nx,nu);
Sh0 = zeros(nx,1);

for k=1:n
    % k0
    [f0, df0_dx, df0_du] = f(x0,        u);
    dSx0 = df0_dx*Sx0;
    dSu0 = df0_dx*Su0 + df0_du;
    dSh0 = df0_dx*Sh0 + f0/h0;

    % k1
    [f1, df1_dx, df1_du] = f(x0 + h*h0*f0/2, u);
    dSx1 = df1_dx*(Sx0 + h*h0/2*dSx0);
    dSu1 = df1_dx*(Su0 + h*h0/2*dSu0) + df1_du;
    dSh1 = df1_dx*(Sh0 + h*h0/2*dSh0) + f1/h0;

    % k2
    [f2, df2_dx, df2_du] = f(x0 + h*h0*f1/2, u);
    dSx2 = df2_dx*(Sx0 + h*h0/2*dSx1);
    dSu2 = df2_dx*(Su0 + h*h0/2*dSu1) + df2_du;
    dSh2 = df2_dx*(Sh0 + h*h0/2*dSh1) + f2/h0;

    % k3
    [f3, df3_dx, df3_du] = f(x0 + h*h0*f2,   u);
    dSx3 = df3_dx*(Sx0 + h*h0*dSx2);
    dSu3 = df3_dx*(Su0 + h*h0*dSu2) + df3_du;
    dSh3 = df3_dx*(Sh0 + h*h0*dSh2) + f3/h0;

    % add em up
    x1 = x0 + (f0 + 2*f1 + 2*f2 + f3)*h*h0/6;
    Sx1 = Sx0 + (dSx0 + 2*dSx1 + 2*dSx2 + dSx3)*h*h0/6;
    Su1 = Su0 + (dSu0 + 2*dSu1 + 2*dSu2 + dSu3)*h*h0/6;
    Sh1 = Sh0 + (dSh0 + 2*dSh1 + 2*dSh2 + dSh3)*h*h0/6;

    % reset for the loop
    x0 = x1;
    Sx0 = Sx1;
    Su0 = Su1;
    Sh0 = Sh1;
end

end

function [x1,Sx1,Su1] = rk4_xu(x0,u,h,n)

h = h/n; % n steps

nx = length(x0);
nu = length(u);

Sx0 = eye(nx);
Su0 = zeros(nx,nu);

for k=1:n
    [f0, df0_dx, df0_du] = f(x0,        u);
    dSx0 = df0_dx*Sx0;
    dSu0 = df0_dx*Su0 + df0_du;
    
    [f1, df1_dx, df1_du] = f(x0+h*f0/2, u);
    dSx1 = df1_dx*(Sx0 + h/2*dSx0);
    dSu1 = df1_dx*(Su0 + h/2*dSu0) + df1_du;
    
    [f2, df2_dx, df2_du] = f(x0+h*f1/2, u);
    dSx2 = df2_dx*(Sx0 + h/2*dSx1);
    dSu2 = df2_dx*(Su0 + h/2*dSu1) + df2_du;
    
    [f3, df3_dx, df3_du] = f(x0+h*f2,   u);
    dSx3 = df3_dx*(Sx0 + h*dSx2);
    dSu3 = df3_dx*(Su0 + h*dSu2) + df3_du;
    
    x1 = x0 + (f0 + 2*f1 + 2*f2 + f3)*h/6;
    Sx1 = Sx0 + (dSx0 + 2*dSx1 + 2*dSx2 + dSx3)*h/6;
    Su1 = Su0 + (dSu0 + 2*dSu1 + 2*dSu2 + dSu3)*h/6;

    % reset for the loop
    x0 = x1;
    Sx0 = Sx1;
    Su0 = Su1;
end

end
