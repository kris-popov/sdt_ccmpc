clear; clc; close all;
import casadi.*

% MPC Parameters
Ts   = 0.10;
N    = 20;
Tsim = 80;
nSCP = 2;


nx = 12; % [px py pz vx vy vz phi theta psi wx wy wz]
nu = 4;  % [T tau_x tau_y tau_z]

% Model Parameters
m  = 1.0;
g  = 9.81;
Jx = 0.02;
Jy = 0.02;
Jz = 0.04;
J  = diag([Jx Jy Jz]);

T_min = 0.2*m*g; % min thrust
T_max = 1.8*m*g; % max thrust
tau_max = [0.25; 0.25; 0.15]; % torque magnitude limits

angle_max = deg2rad(35); % Euler-angle limits

r_robot = 0.25; % quadrotor radius

% gaussian noise
sigma_acc = [0.25; 0.25; 0.18];
Wacc = diag(sigma_acc.^2);

alpha = 0.05;
beta_cc = gauss_quantile(1 - alpha);

Q  = diag([30 30 35, 4 4 5, 2 2 2, 0.2 0.2 0.2]);
Qf = diag([80 80 100, 8 8 10, 4 4 4, 0.5 0.5 0.5]);
R  = diag([0.08, 0.5, 0.5, 0.5]);

u_hover = [m*g; 0; 0; 0]; % nominal eqbrm input

K_tube = lqr_gain(Ts, m, g, J, Q, R);

D = zeros(nx,3);
D(4:6,:) = eye(3);

% Box obstacles: (per row) [xmin xmax ymin ymax zmin zmax]
% To run each case, comment/uncomment the others

% Case 1:

x0 = zeros(nx,1);
x0(1:3) = [0; 0; 1.0];

x_goal = zeros(nx,1);
x_goal(1:3) = [8.5; 8.0; 2.0];

pos_min = [-1; -1; 0.2];
pos_max = [10; 10; 4.0];

boxes = [
    2.5 4.0   2.0 3.5   0.5 4.0;
    5.6 7.0   3.4 4.8   0.5 2.6;
    3.2 4.8   6.1 7.6   0.7 3.0
];

% Case 2:

% x0 = zeros(nx,1);
% x0(1:3) = [0; 0; 2.0];
%
% x_goal = zeros(nx,1);
% x_goal(1:3) = [9.0; 0.0; 2.0];
%
% pos_min = [-1; -1; 0.0];
% pos_max = [10.0; 1; 4.0];
%
% boxes = [
%     2.0 3.5   -1.0 1.0   0.0 2.0;
%     5.6 7.0   -1.0 1.0   2.0 4.0
% ];

% Case 3:

% x0 = zeros(nx,1);
% x0(1:3) = [0; 0; 0.5];
%
% x_goal = zeros(nx,1);
% x_goal(1:3) = [6.0; 0.0; 3.5];
%
% pos_min = [-1; -1; 0.0];
% pos_max = [7.0; 1; 4.0];
%
% boxes = [
%     2.5 3.5   -1.0 1.0   0.0 1.5;
%     2.5 3.5   -1.0 1.0   2.5 4.0;
%     2.5 3.5   -1.0 -0.5  0.0 4.0;
%     2.5 3.5   0.5 1.0    0.0 4.0
% ];

nObs = size(boxes,1);

obs_eps = 1e-3;

rng(2);

% CasADi setup
x_sym = SX.sym('x', nx);
u_sym = SX.sym('u', nu);

xdot_sym = quad_dynamics_euler_sx(x_sym, u_sym, m, g, J);
f_ode = Function('f_ode', {x_sym, u_sym}, {xdot_sym});

k1 = f_ode(x_sym, u_sym);
k2 = f_ode(x_sym + Ts/2*k1, u_sym);
k3 = f_ode(x_sym + Ts/2*k2, u_sym);
k4 = f_ode(x_sym + Ts*k3, u_sym);

x_next_sym = x_sym + Ts/6*(k1 + 2*k2 + 2*k3 + k4);

f_disc = Function('f_disc', {x_sym, u_sym}, {x_next_sym});

A_sym = jacobian(x_next_sym, x_sym);
B_sym = jacobian(x_next_sym, u_sym);

f_A = Function('f_A', {x_sym, u_sym}, {A_sym});
f_B = Function('f_B', {x_sym, u_sym}, {B_sym});

% initilize solver
x_actual = x0;

X_hist = zeros(nx,Tsim+1);
U_hist = zeros(nu,Tsim);
X_hist(:,1) = x_actual;

X_guess = zeros(nx,N+1);
U_guess = repmat(u_hover,1,N);

for k = 1:N+1
    a = (k-1)/N;
    X_guess(:,k) = (1-a)*x0 + a*x_goal;
end

% Simulate and solve
for t = 1:Tsim

    x_nom0 = x_actual;

    for scp = 1:nSCP

        Sigma_seq = propagate_covariance( ...
            f_A, f_B, X_guess, U_guess, K_tube, D, Wacc, N);

        [X_sol, U_sol, success] = solve_quadrotor_nmpc_box_sdf( ...
            f_disc, x_nom0, x_goal, X_guess, U_guess, ...
            Sigma_seq, boxes, r_robot, beta_cc, obs_eps, ...
            Q, Qf, R, u_hover, ...
            T_min, T_max, tau_max, pos_min, pos_max, angle_max, ...
            N, nx, nu);

        if ~success
            warning('NMPC failed at sim step %d, SCP iter %d.', t, scp);
            X_sol = X_guess;
            U_sol = U_guess;
            break;
        end

        X_guess = X_sol;
        U_guess = U_sol;
    end

    u_nom = U_sol(:,1);

    e = x_actual - X_sol(:,1);
    e(7:9) = wrapToPi(e(7:9));

    u_applied = u_nom + K_tube*e;

    u_applied(1) = min(max(u_applied(1), T_min), T_max);
    u_applied(2:4) = min(max(u_applied(2:4), -tau_max), tau_max);

    x_next = full(f_disc(x_actual, u_applied));

    w_acc = chol(Wacc,'lower') * randn(3,1);
    x_next(4:6) = x_next(4:6) + Ts*w_acc;
    x_next(7:9) = wrapToPi(x_next(7:9));

    X_hist(:,t+1) = x_next;
    U_hist(:,t) = u_applied;

    x_actual = x_next;

    X_guess = [X_sol(:,2:end), X_sol(:,end)];
    U_guess = [U_sol(:,2:end), U_sol(:,end)];

    if norm(x_actual(1:3)-x_goal(1:3)) < 0.25 && norm(x_actual(4:6)) < 0.25
        X_hist = X_hist(:,1:t+1);
        U_hist = U_hist(:,1:t);
        fprintf('Reached goal at step %d.\n', t);
        break;
    end
end

% DEBUG output, distnace
min_clearance = inf;
for k = 1:size(X_hist,2)
    p = X_hist(1:3,k);
    for j = 1:nObs
        [d,~] = box_sdf_and_grad(p, boxes(j,:)');
        min_clearance = min(min_clearance, d - r_robot);
    end
end
fprintf('Minimum trajectory-clearance: %.4f m\n', min_clearance);




% Plot
figure('Color','w'); hold on; grid on; axis equal;
xlabel('x'); ylabel('y'); zlabel('z');

plot3(X_hist(1,:), X_hist(2,:), X_hist(3,:), 'b-', 'LineWidth', 2.2);
plot3(x0(1), x0(2), x0(3), 'go', 'MarkerFaceColor','g', 'MarkerSize', 9);
plot3(x_goal(1), x_goal(2), x_goal(3), 'mp', 'MarkerFaceColor','m', 'MarkerSize', 12);

for j = 1:nObs
    plot_box(boxes(j,:), [0.85 0.2 0.2], 0.35);
end

xlim([pos_min(1), pos_max(1)]);
ylim([pos_min(2), pos_max(2)]);
zlim([pos_min(3), pos_max(3)]);
view(35,25);

figure('Color','w');
subplot(4,1,1); plot(U_hist(1,:), 'LineWidth',1.5); grid on; ylabel('T');
subplot(4,1,2); plot(U_hist(2,:), 'LineWidth',1.5); grid on; ylabel('\tau_x');
subplot(4,1,3); plot(U_hist(3,:), 'LineWidth',1.5); grid on; ylabel('\tau_y');
subplot(4,1,4); plot(U_hist(4,:), 'LineWidth',1.5); grid on; ylabel('\tau_z'); xlabel('step');




% Functions

function [X_sol, U_sol, success] = solve_quadrotor_nmpc_box_sdf( ...
    f_disc, x0, x_goal, X_guess, U_guess, ...
    Sigma_seq, boxes, r_robot, beta_cc, obs_eps, ...
    Q, Qf, R, u_hover, ...
    T_min, T_max, tau_max, pos_min, pos_max, angle_max, ...
    N, nx, nu)

    import casadi.*

    X = SX.sym('X', nx, N+1);
    U = SX.sym('U', nu, N);

    obj = 0;
    gcon = {};
    lbg = [];
    ubg = [];

    gcon{end+1} = X(:,1) - x0;
    lbg = [lbg; zeros(nx,1)];
    ubg = [ubg; zeros(nx,1)];

    for k = 1:N

        xk = X(:,k);
        uk = U(:,k);

        gcon{end+1} = X(:,k+1) - f_disc(xk, uk);
        lbg = [lbg; zeros(nx,1)];
        ubg = [ubg; zeros(nx,1)];

        dx = xk - x_goal;
        du = uk - u_hover;
        obj = obj + dx'*Q*dx + du'*R*du;

        gcon{end+1} = xk(1:3);
        lbg = [lbg; pos_min];
        ubg = [ubg; pos_max];

        gcon{end+1} = xk(7:9);
        lbg = [lbg; -angle_max*ones(3,1)];
        ubg = [ubg;  angle_max*ones(3,1)];

        p_ref = X_guess(1:3,k);
        Sig_p = Sigma_seq(1:3,1:3,k);

        for j = 1:size(boxes,1)
            box = boxes(j,:)';

            [d_ref, grad_d] = box_sdf_and_grad(p_ref, box);

            sigma_sdf = sqrt(max(grad_d' * Sig_p * grad_d, 0));

            sdf_lin = d_ref + grad_d'*(xk(1:3) - p_ref);

            gcon{end+1} = sdf_lin;
            lbg = [lbg; r_robot + beta_cc*sigma_sdf + obs_eps];
            ubg = [ubg; inf];
        end
    end

    dxN = X(:,N+1) - x_goal;
    obj = obj + dxN'*Qf*dxN;

    gcon{end+1} = X(1:3,N+1);
    lbg = [lbg; pos_min];
    ubg = [ubg; pos_max];

    z = [reshape(X, nx*(N+1), 1);
         reshape(U, nu*N, 1)];

    G_all = vertcat(gcon{:});

    nlp = struct('x', z, 'f', obj, 'g', G_all);

    opts = struct;
    opts.ipopt.print_level = 0;
    opts.print_time = false;
    opts.ipopt.max_iter = 300;
    opts.ipopt.tol = 1e-4;
    opts.ipopt.acceptable_tol = 1e-3;

    solver = nlpsol('solver','ipopt',nlp,opts);

    lbz = -inf(size(z));
    ubz =  inf(size(z));

    offset_U = nx*(N+1);

    for k = 1:N
        idx = offset_U + (k-1)*nu + (1:nu);

        lbz(idx(1)) = T_min;
        ubz(idx(1)) = T_max;

        lbz(idx(2:4)) = -tau_max;
        ubz(idx(2:4)) =  tau_max;
    end

    z0 = [reshape(X_guess, nx*(N+1), 1);
          reshape(U_guess, nu*N, 1)];

    try
        sol = solver('x0', z0, 'lbx', lbz, 'ubx', ubz, 'lbg', lbg, 'ubg', ubg);
        zsol = full(sol.x);

        X_vec = zsol(1:nx*(N+1));
        U_vec = zsol(nx*(N+1)+1:end);

        X_sol = reshape(X_vec, nx, N+1);
        U_sol = reshape(U_vec, nu, N);

        success = true;
    catch
        X_sol = X_guess;
        U_sol = U_guess;
        success = false;
    end
end

function Sigma_seq = propagate_covariance(f_A, f_B, Xref, Uref, K, G, W, N)
    nx = size(Xref,1);
    Sigma_seq = zeros(nx,nx,N+1);
    Sigma_seq(:,:,1) = zeros(nx);

    for k = 1:N
        Ak = full(f_A(Xref(:,k), Uref(:,k)));
        Bk = full(f_B(Xref(:,k), Uref(:,k)));

        Acl = Ak + Bk*K;

        Sigma_seq(:,:,k+1) = Acl*Sigma_seq(:,:,k)*Acl' + G*W*G';
        Sigma_seq(:,:,k+1) = 0.5*(Sigma_seq(:,:,k+1)+Sigma_seq(:,:,k+1)');
    end
end

function [d, grad] = box_sdf_and_grad(p, box)
    p = p(:);

    xmin = box(1); xmax = box(2);
    ymin = box(3); ymax = box(4);
    zmin = box(5); zmax = box(6);

    c = [(xmin+xmax)/2; (ymin+ymax)/2; (zmin+zmax)/2];
    h = [(xmax-xmin)/2; (ymax-ymin)/2; (zmax-zmin)/2];

    q = abs(p - c) - h;

    q_pos = max(q, 0);
    outside_dist = norm(q_pos);
    inside_dist = min(max(q(1), max(q(2), q(3))), 0);

    d = outside_dist + inside_dist;

    if outside_dist > 1e-9
        closest = min(max(p, [xmin; ymin; zmin]), [xmax; ymax; zmax]);
        v = p - closest;
        grad = v / norm(v);
    else
        dist_to_faces = [
            abs(p(1)-xmin);
            abs(xmax-p(1));
            abs(p(2)-ymin);
            abs(ymax-p(2));
            abs(p(3)-zmin);
            abs(zmax-p(3))
        ];

        [~,idx] = min(dist_to_faces);

        switch idx
            case 1
                grad = [-1;0;0];
            case 2
                grad = [1;0;0];
            case 3
                grad = [0;-1;0];
            case 4
                grad = [0;1;0];
            case 5
                grad = [0;0;-1];
            case 6
                grad = [0;0;1];
        end
    end
end

function xdot = quad_dynamics_euler_sx(x, u, m, g, J)
    import casadi.*

    vx = x(4); vy = x(5); vz = x(6);

    phi   = x(7);
    theta = x(8);
    psi   = x(9);

    wx = x(10);
    wy = x(11);
    wz = x(12);

    T = u(1);
    tau = u(2:4);

    R = Rzyx_sx(phi, theta, psi);

    e3 = [0;0;1];

    p_dot = [vx; vy; vz];
    v_dot = (T/m) * R * e3 - [0;0;g];

    omega = [wx; wy; wz];
    omega_dot = inv(SX(J)) * (tau - cross_sx(omega, SX(J)*omega));

    E = euler_rate_matrix_sx(phi, theta);
    eul_dot = E * omega;

    xdot = [p_dot; v_dot; eul_dot; omega_dot];
end

function R = Rzyx_sx(phi, theta, psi)
    cphi = cos(phi);   sphi = sin(phi);
    cth  = cos(theta); sth  = sin(theta);
    cpsi = cos(psi);   spsi = sin(psi);

    Rz = [cpsi -spsi 0;
          spsi  cpsi 0;
          0     0    1];

    Ry = [cth 0 sth;
          0   1 0;
         -sth 0 cth];

    Rx = [1 0 0;
          0 cphi -sphi;
          0 sphi  cphi];

    R = Rz*Ry*Rx;
end

function E = euler_rate_matrix_sx(phi, theta)
    E = [1, sin(phi)*tan(theta), cos(phi)*tan(theta);
         0, cos(phi),            -sin(phi);
         0, sin(phi)/cos(theta), cos(phi)/cos(theta)];
end

function c = cross_sx(a,b)
    c = [a(2)*b(3)-a(3)*b(2);
         a(3)*b(1)-a(1)*b(3);
         a(1)*b(2)-a(2)*b(1)];
end

function K = lqr_gain(Ts, m, g, J, Q, R)
    nx = 12;
    nu = 4;

    Ac = zeros(nx,nx);
    Bc = zeros(nx,nu);

    Ac(1,4) = 1;
    Ac(2,5) = 1;
    Ac(3,6) = 1;

    Ac(4,8) = g;
    Ac(5,7) = -g;
    Bc(6,1) = 1/m;

    Ac(7,10) = 1;
    Ac(8,11) = 1;
    Ac(9,12) = 1;

    Bc(10,2) = 1/J(1,1);
    Bc(11,3) = 1/J(2,2);
    Bc(12,4) = 1/J(3,3);

    Ad = eye(nx) + Ts*Ac;
    Bd = Ts*Bc;

    [Kdlqr,~,~] = dlqr(Ad, Bd, Q, R);

    K = -Kdlqr;
end

function q = gauss_quantile(p)
    q = -sqrt(2) * erfcinv(2*p);
end

function y = wrapToPi(x)
    y = mod(x + pi, 2*pi) - pi;
end

function plot_box(box, color, alpha_val)
    xmin = box(1); xmax = box(2);
    ymin = box(3); ymax = box(4);
    zmin = box(5); zmax = box(6);

    V = [
        xmin ymin zmin;
        xmax ymin zmin;
        xmax ymax zmin;
        xmin ymax zmin;
        xmin ymin zmax;
        xmax ymin zmax;
        xmax ymax zmax;
        xmin ymax zmax
    ];

    F = [
        1 2 3 4;
        5 6 7 8;
        1 2 6 5;
        2 3 7 6;
        3 4 8 7;
        4 1 5 8
    ];

    patch('Vertices', V, 'Faces', F, ...
          'FaceColor', color, ...
          'FaceAlpha', alpha_val, ...
          'EdgeColor', 'k', ...
          'LineWidth', 1.0);
end
