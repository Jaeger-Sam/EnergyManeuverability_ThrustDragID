% ThrustDragID_EM.fcn finds the thrust and drag coefficient parameters
% for a given time series of data using energy maneuverability. The default
% option is least-squares but the user has the option to fix certain
% variables, perform stepwise regression (in a least-squares sense), and
% run an iterative convex optimization problem for a range of sparsity 
% hyperparameters. The stepwise regression and iterative convex 
% optimization problem are tools to find the thrust and drag model form.
% Mosek must be installed to use the convex optimization features. User
% must input data with appropriate units, remove biases and significant 
% measurement noise. This function will call LP_15smooth.fcn.
%
% [theta_hat, E_dot_s, Jc_opt, Q_theta, P_s, theta_hat_kappa_grid, Jci, Jc_1, Jc_2] = ThrustDragID_EM(dt,u,alpha,beta,a_xyz,delta_vec,n,rho,d,W,Sw,g,param,param_fix, V_regress, sig_sensor, stepwise_regress, convex_sol)
%
% INPUTS:
%   dt: sample timestep (s), scalar
%   u: airspeed (ft/s or m/s), vector of length Nx1
%   alpha: angle of attack (rad), vector of length Nx1
%   beta: angle of sideslip (rad), vector of length Nx1
%   a_xyz: acceleration in body frame (ft/s^2 or m/s^2), matrix of size Nx3
%   delta_vec: control surface deflections (rad), matrix of size Nx4
%       with columns in {delta_e, delta_a, delta_r, delta_f} order
%   n: propeller rotation rate (rev/s), vector of Nx1
%   rho: air density (slugs/ft^3 or kg/m^3), scalar
%   d: propeller diameter (ft or m), scalar
%   W: weight (lb or N), scalar
%   Sw: wing reference area (ft^2 or m^2), scalar
%   g: acceleration due to gravity (ft/s^2 or ft/s^2), scalar
%   param: row vector of true / false with parameters to be fit in order...
%     {CT2 CT1 CT0 CD0 CDa CDa2 CDb CDb2 CDde2 CDda2 CDdr2 CDdf2 CDv}
%   param_fix: row vector of parameters that are to be fixed in order...
%     {CT2 CT1 CT0 CD0 CDa CDa2 CDb CDb2 CDde2 CDda2 CDdr2 CDdf2 CDv}
%      If a particular parameter is to be identified, pass that value in 
%      as NaN. If a numerical value is inputted in param_fix and
%      the associated parameter is also true in param, program will error.
%
% OPTIONAL INPUTS:
%   V_regress: true / false to format H matrix with V/W factor in all 
%       regressors (default true).
%   sig_sensor: vector of standard deviations of each of the sensors used
%       in the sensed energy maneuverability function (must follow the same
%       units as the inputs above). If an empty variable is input, the
%       program will try and estimate these parameters using a discrete
%       time low pass filter.
%       {sig_u, sig_ax, sig_ay, sig_az, sig_alpha, sig_beta_f}
%   stepwise_regress: true / false to perform stepwise regression. Will ask
%       for user input to automate or have user remove variables manually.
%   convex_sol: true / false to perform convex optimization solution 
%       (must have Mosek installed). Will ask user to either fix the
%       sparsity hyperparameter or grid on a range of hyperparameters.
%
% OUTPUTS:
%   theta_hat: vector of estimated parameters with fixed an
%   E_dot_s: measured energy maneuverability time history
%   Jc_opt: 2-norm of error between fitted and sensed energy
%       maneuverability
%
% OPTIONAL OUTPUTS FROM UNCERTAINTY ANALYSIS: 
%   Q_theta: covariance matrix of estimated parameters 
%       (empty if V_regress == false)
%   P_s: reconstructed energy maneuverability time history
%       (empty if V_regress == false)
% OPTIONAL OUTPUTS FROM ITERATIVE CONVEX OPTIMIZATION SOLUTION:
%   theta_hat_kappa_grid: matrix of solutions for parameter estimates using
%       the iterative convex solution (columns are a particular sparsity 
%       value).
%   Jci: matrix of cost function values at for each parameter at 
%       each level of sparsity (rows are parameters, columns are sparsity 
%       values).
%   Jc_1: 1-norm portion of the objective function for a range of sparsity
%   Jc_2: 2-norm portion of the objective function for a range of sparsity
%
% See publication for more information...
% "Parametric Thrust and Drag Identification from Flight Test via Energy 
% Maneuverability" by Sam Jaeger, et. al., 2026 AIAA Aviation Forum,
% DOI: 10.2514/6.2026-416
%
% Sam Jaeger
% 6/22/2026
% jaege246@umn.edu

function [theta_hat, E_dot_s, Jc_opt, varargout] = ThrustDragID_EM(dt,u,alpha,beta,a_xyz,delta_vec,n,rho,d,W,Sw,g,param,param_fix, varargin)
    % input logic ---------------------------------------------------------
    narginchk(14,18)
    n_arguments = length(varargin);

    if n_arguments == 1
        V_regress = varargin{1}; % choose if regressors are formatted with V or not
        sig_sensor = [];
        stepwise_regress = false;
        convex_sol = false;

    elseif n_arguments == 2
        V_regress = varargin{1};
        sig_sensor = varargin{2};
        stepwise_regress = false;
        convex_sol = false;

    elseif n_arguments == 3
        V_regress = varargin{1};
        sig_sensor = varargin{2};
        stepwise_regress = varargin{3};
        convex_sol = false;

    elseif n_arguments == 4
        V_regress = varargin{1};
        sig_sensor = varargin{2};
        stepwise_regress = varargin{3};
        convex_sol = varargin{4};

    else  % default
        V_regress = true;
        sig_sensor = [];
        stepwise_regress = false;
        convex_sol = false;
    end

    [N,JJ] =size(u);
    if JJ ~= 1 
        error('u input must be a column vector of length N x 1')
    end
    [II,JJ] = size(alpha);
    if II ~= N || JJ ~= 1
        error('alpha input must be a column vector of length N x 1')
    end
    [II,JJ] = size(beta);
    if II ~= N || JJ ~= 1
        error('beta input must be a column vector of length N x 1')
    end
    [II,JJ] = size(a_xyz);
    if II ~= N || JJ ~= 3
        error('a_xyz input must be a matrix of size N x 3')
    end
    [II,JJ] = size(delta_vec);
    if II ~= N || JJ ~= 4
        error('delta_vec input must be a matrix of size N x 4')
    end
    [II,JJ] = size(n);
    if II ~= N || JJ ~= 1
        error('n input must be a column vector of length N x 1')
    end
    [MM,KK] = size(param);
    if MM ~= 1 || KK ~= 13
        error('param input must be a row vector of length 1 x 13 ')
    end
    [MM,KK] = size(param_fix);
    if MM ~= 1 || KK ~= 13
        error('param_fix input must be a row vector of length 1 x 13 ')
    end

    for ii=1:length(param)
        if param(ii) == 1 && isnan(param_fix(ii)) == false
            error('Cannot both identify and fix parameter! Change param and param_fix inputs.')
        end
    end

    % assemble regressors -------------------------------------------------
    CTab = cos(alpha).*cos(beta); % angle correction factor
    V = u./CTab; % correct total velocity
    qbar = 0.5*rho*V.^2; % dynamic pressure 
    J = V./n./d; % Advance ratio
    
    H_CT2 = CTab.*(J.^2).*rho.*(n.^2).*(d^4); % CT,J^2
    H_CT1 = CTab.*(J).*rho.*(n.^2).*(d^4); % CT,J
    H_CT0 = CTab.*rho.*(n.^2).*(d^4); % CT0
    H_CD0 = -qbar*Sw; % CD0 
    H_alpha = -qbar.*Sw.*(alpha); % CD,alpha^2
    H_alpha2 = -qbar.*Sw.*(alpha.^2); % CD,alpha^2
    H_beta = -qbar.*Sw.*(beta); % CD,beta^2
    H_beta2 = -qbar.*Sw.*(beta.^2); % CD,beta^2
    H_de2 = -qbar.*Sw.*(delta_vec(:,1).^2); % CD,delta_e^2
    H_da2 = -qbar.*Sw.*(delta_vec(:,2).^2); % CD,delta_a^2
    H_dr2 = -qbar.*Sw.*(delta_vec(:,3).^2); % CD,delta_r^2
    H_df2 = -qbar.*Sw.*(delta_vec(:,4).^2); % CD,delta_f^2
    H_CDv = -qbar.*Sw.*(V); % CD,V

    % legend
    var_reg = {'$C_{T_2}$','$C_{T_1}$','$C_{T_0}$','$C_{D_0}$','$C_{D,\alpha}$','$C_{D,\alpha_2}$','$C_{D,\beta}$','$C_{D,\beta_2}$','$C_{D,{\delta_e}_2}$','$C_{D,{\delta_a}_2}$','$C_{D,{\delta_r}_2}$','$C_{D,{\delta_f}_2}$','$C_{D,{V}}$'};


    % full H matrix
    if V_regress == true
        H_full = V./W.*[H_CT2,H_CT1,H_CT0,H_CD0,H_alpha,H_alpha2,H_beta,H_beta2,H_de2,H_da2,H_dr2,H_df2,H_CDv];
    else    
        H_full = [H_CT2,H_CT1,H_CT0,H_CD0,H_alpha,H_alpha2,H_beta,H_beta2,H_de2,H_da2,H_dr2,H_df2,H_CDv];
    end

    % remove columns based on parameters to be fit
    H = [];
    count = 1;
    for ii=1:length(param)
        if param(ii) == 1
            H(:,count) = H_full(:,ii);
            count = count + 1;
        end
    end

    % Sensed Energy Maneuverability
    beta_f = atan2(tan(beta),cos(alpha)); % flank angle (rad)
    E_dot_s = u/g.*(a_xyz(:,1) + tan(beta_f).*a_xyz(:,2) + tan(alpha).*a_xyz(:,3));

    % RHS of linear system of eqns
    if V_regress == true
        y = E_dot_s;
    else
        m = W/g;
        y = m*(a_xyz(:,1) + tan(beta_f).*a_xyz(:,2) + tan(alpha).*a_xyz(:,3));
    end

    % removed fixed parameters from RHS of linear system of equation
    for ii=1:length(param_fix)
        if isnan(param_fix(ii)) == false
            y = y - param_fix(ii)*H_full(:,ii);
        end
    end

    % least squares solution ----------------------------------------------
    disp('===================================')
    disp('===== LEAST SQUARES SOLUTION ======')
    theta_hat = (H'*H)\H'*y;
    y_recon = H*theta_hat;
    Jc_opt = norm((y_recon - y),2);
    count = 1;
    for ii=1:length(param)
        if param(ii) == 1
            disp(append(string(var_reg(ii)),' = ',num2str(theta_hat(count)) ) )
            count = count+1;
        end
    end
    disp(append('---------- Jc = ',num2str(Jc_opt),'-----------'))

    % Uncertainty analysis ------------------------------------------------
    if V_regress == true
        if isempty(sig_sensor) == false
            % standard deviations - from user
            du = sig_sensor(1);
            dax = sig_sensor(2);
            day = sig_sensor(3);
            daz = sig_sensor(4);
            dalpha = sig_sensor(5); % rad;
            dbeta_f = sig_sensor(6); % rad
       else
            % estimated standard deviations 
            du = std(LP_15smooth(u) - u);
            dax = std(LP_15smooth(a_xyz(:,1)) - a_xyz(:,1));
            day = std(LP_15smooth(a_xyz(:,2)) - a_xyz(:,2));
            daz = std(LP_15smooth(a_xyz(:,3)) - a_xyz(:,3));
            dalpha = std(LP_15smooth(alpha) - alpha);
            dbeta_f = std(LP_15smooth(beta_f) - beta_f);
        end
    
        % sensitivities
        dEdot_du = (a_xyz(:,1) + tan(beta_f).*a_xyz(:,2) + tan(alpha).*a_xyz(:,3))./g;
        dEdot_dax = u/g;
        dEdot_day = (u/g).*tan(beta_f);
        dEdot_daz = (u/g).*tan(alpha);
        dEdot_dalpha = (u/g).*(sec(alpha).^2).*a_xyz(:,3);
        dEdot_dbeta_f = (u/g).*(sec(beta_f).^2).*a_xyz(:,2);
        
        % combined sensitivities
        d_u = (dEdot_du.*du).^2;
        d_ax = (dEdot_dax.*dax).^2;
        d_ay = (dEdot_day.*day).^2;
        d_az = (dEdot_daz.*daz).^2;
        d_alpha = (dEdot_dalpha.*dalpha).^2;
        d_beta_f = (dEdot_dbeta_f.*dbeta_f).^2;
        
        d_Edot_s = sqrt( d_u +  d_ax + d_ay + d_az + d_alpha + d_beta_f );
    
        R_EM = diag(d_Edot_s.^2);
   
        % covariance
        Q_theta = inv(H'*inv(R_EM)*H);

        % reconstruct EM
        H_ID = []; 
        theta_ID_0 = [];
        count = 1;
        count_paramID = 1; 
        for ii=1:length(param)
            if param(ii) == true
                H_ID(:,count) = H_full(:,ii);
                theta_ID_0(count,1) = theta_hat(count_paramID);
                count = count+1;
                count_paramID = count_paramID+1;
            elseif isnan(param_fix(ii)) == false
                H_ID(:,count) = H_full(:,ii);
                theta_ID_0(count,1) = param_fix(ii);
                count = count+1;
            end
        end
        P_s = H_ID*theta_ID_0;

        t = linspace(0,dt*N,N);
        figure(282)
        plot(t,E_dot_s,'o'); hold on;
        plot(t,P_s,'.','color',[0.2310,0.6660,0.1960]); 
        plot(t,E_dot_s + 3*d_Edot_s,'color',[0.5,0.5,0.5]);
        plot(t,E_dot_s - 3*d_Edot_s,'color',[0.5,0.5,0.5]); grid on; 
        xlabel('time (s)','Interpreter','latex','FontSize',20)
        ylabel('$P_s$ $(ft/s)$','Interpreter','latex','FontSize',20)
        legend('Measured','Reconstructed','$\pm 3\sigma$','Interpreter','latex','FontSize',12,'Location','best')
        hold off
    else
        Q_theta = [];
        P_s = [];
    end
    varargout{1} = Q_theta;
    varargout{2} = P_s;


    % Stepwise Regression -------------------------------------------------
    if stepwise_regress == true
        % disp('=========================================')
        % disp('=========================================')
        disp('=========================================')
        disp('===== Starting Stepwise regression ======')
                
        tf_auto_stepwise_reg = input('Automate Stepwise Regression (true / false)?');
        
        H_var = param;
        for ii=1:sum(param)
            disp('=========================================')
            disp(append('Iteration = ',num2str(ii)))
        
            
            count = 1;
            H = [];
            for ii=1:length(H_var)
                if H_var(ii) == 1
                    H(:,count) = H_full(:,ii);
                    count = count + 1;
                end
            end
            theta_hat = (H'*H)\H'*y;
            y_recon = H*theta_hat;
            count = 1;
            Jci = [];
            for ii=1:length(H_var)
                if H_var(ii) == 1
                    disp('-------------------------------------')
                    Jci(ii) = norm( H(:,count)*theta_hat(count) - y,2);
                    disp(append(string(var_reg(ii)),' = ',num2str(theta_hat(count)) ) )
                    disp(append(string(var_reg(ii)),', ii =',num2str(ii),', Jci = ',num2str(Jci(ii)) ) )
                    count = count +1;
                else
                    Jci(ii) = NaN;
                end
            end
            t = linspace(0,dt*N,N);
            y_recon = H*theta_hat;

            figure(182)
            plot( t,y,'.'); hold on
            plot( t,y_recon,'.');  grid on
            xlabel('time (s)','Interpreter','latex','FontSize',20)
            ylabel('$y$','Interpreter','latex','FontSize',20)
            legend('Measured','Reconstructed','Interpreter','latex','Location','northwest','FontSize',12)
            hold off

            tf_stop_reg = input("Stop regression? (true / false)");
            if tf_stop_reg == true
                disp('=========================================')
                disp('Stopping regression')
                disp('Least-squares solution...')

                count = 1;
                for ii=1:length(H_var)
                    if H_var(ii) == 1
                        disp(append(string(var_reg(ii)),' = ',num2str(theta_hat(count)) ) )
                        count = count+1;
                    end
                end
                disp(append('---------- Jc = ',num2str(Jc_opt),'-----------'))
                break
            end
            if tf_auto_stepwise_reg == true
                clc;
                [~,i_delete]=min(Jci);
                disp('automated deleting....')
                disp(append(string(var_reg(i_delete)),', ii =',num2str(i_delete),', Jci = ',num2str(Jci(i_delete)) ) )
            else
                i_delete = input("Index to delete? ii =");
                clc;
            end
            H_var(i_delete) = 0;
            close(182)
            
        end
        
       
    end

    % convex optimization -------------------------------------------------
    if convex_sol == true
        disp('====================================')
        disp('== STARTING CONVEX OPTIMIZATION ====')
        n_var = size(H,2);
        grid_on_kappa_tf = input('Grid on sparsity parameter (true / false)?');

            %               1   2   3   4   5    6   7    8     9    10    11   12   13
            %             CT2 CT1 CT0 CD0 CDa CDa2 CDb CDb2 CDde2 CDda2 CDdr2 CDdf2  CDv
            theta_lower = [-1,  -1,  0,  0, -1,   0, -1,  0,    0,    0,    0,    0,  0]';
            theta_upper = [  0,  0,  1,  1,  5,   5,  1,  1,    1,    1,    1,    1, 10]';
            
            count = 1;
            H = [];
            theta_lb = [];
            theta_ub = [];
            for ii=1:length(param)
                if param(ii) == 1
                    H(:,count) = H_full(:,ii);
                    theta_lb(count,1) = theta_lower(ii);
                    theta_ub(count,1) = theta_upper(ii);
                    count = count + 1;
                end
            end

        if grid_on_kappa_tf == false
            kappa_fix = input('Input fixed sparsity parameter (must be greater than zero) = ');
            
            options = sdpsettings('verbose',0,'solver','mosek');
            theta_hat_c = sdpvar(n_var,1);
            e = H*theta_hat_c - y;
            constraints =  [theta_lb <= theta_hat_c <= theta_ub];
            optimize(constraints,norm(e)^2 + kappa_fix*norm(theta_hat_c,1),options);
            theta_hat = value(theta_hat_c);
            
            disp('====================================')
            disp('======= CONVEX LS + L1 SOLUTION ====')
            count = 1;
            for ii=1:length(param)
                if param(ii) == 1
                    disp(append(string(var_reg(ii)),' = ',num2str(theta_hat(count)) ) )
                    count = count + 1;
                end
            end
            y_recon = H*theta_hat;
            Jc_opt = norm(y_recon - y);
            disp(append('---------- Jc = ',num2str(Jc_opt),'-----------'))
            
            t = linspace(0,dt*N,N);
            figure(182)
            plot( t,y,'.'); hold on
            plot( t,y_recon,'.');  grid on
            xlabel('time (s)','Interpreter','latex','FontSize',20)
            ylabel('$y$','Interpreter','latex','FontSize',20)
            legend('measured','Reconstructed','Interpreter','latex','Location','northwest','FontSize',12)
            hold off
        else
            kappa_end = input('Ending sparsity hyperparameter value (must be positive) =');
            N_kappa = input('Number of sparity hyperparameter to grid on (must be a positive integer) = ');

            options = sdpsettings('verbose',0,'solver','mosek');
            kappa = linspace(0, kappa_end, N_kappa);
            theta_hat_kappa_grid = zeros(n_var,length(kappa));
            Jc_2 = zeros(length(kappa),1);
            Jc_1 = zeros(length(kappa),1);
            for ii=1:length(kappa)
                theta_hat_c = sdpvar(n_var,1);
                e = H*theta_hat_c - y;
                constraints = [theta_lb <= theta_hat_c <= theta_ub];
                optimize(constraints,norm(e,2)^2 + kappa(ii)*norm(theta_hat_c,1),options);
                theta_hat_L2_IRLS = value(theta_hat_c);
                Jc_2(ii) = norm(H*theta_hat_L2_IRLS - y,2);
                Jc_1(ii) = norm(theta_hat_L2_IRLS,1);
                theta_hat_kappa_grid(:,ii) = theta_hat_L2_IRLS;
                for jj=1:length(theta_hat_L2_IRLS)
                    Jci(ii,jj) = norm(H(:,jj)*theta_hat_L2_IRLS(jj) - y,2)^2;
                end
                disp(append('Completed kappa = ',num2str(kappa(ii))))
            end

            count = 1;
            leg =[];
            for ii=1:length(param)
                if param(ii) == 1
                    leg{count} = string(var_reg(ii));
                    count = count + 1;
                end
            end
            
            figure(210); hold on
            col = orderedcolors('gem12');
            for ii=1:n_var
                plot(kappa,(Jci(:,ii)./Jci(1,ii)),'*','Color',col(ii,:))
            end
            grid on;
            legend(leg,'Interpreter','latex','FontSize',18,'Location','eastoutside')
            xlabel('$\kappa$','Interpreter','latex','FontSize',20)
            ylabel('$ J_{c_i,\kappa}/J_{c_i,0} $','Interpreter','latex','FontSize',20)
            hold off

            varargout{3} = theta_hat_kappa_grid;
            varargout{4} = Jci;
            varargout{5} = Jc_1;
            varargout{6} = Jc_2;
        end
    end
end
