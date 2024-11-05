function [exitflag] = RBDO_Kriging

% RBDO_Kriging_v1 created by Chao Hu on March 27, 2022
% This RBDO code is a compact code for reliability-based design
% optimization (RBDO) which uses kriging (Gaussian process regression)
% method for reliability analysis under uncertainty. One mathematical
% example is solved for demonstration purpose.
%
% VARIABLE DEFINITION
% nc: number of constraints ; nv: number of design variables
% rt: target reliability; d0: initial design; dp: previous design
% lb & ub: lower bound and upper bound for design variables
% iter: variable used to record iteration number of optimization process
% UNIFSEED: uniform random seeds for reliability analysis using GP
% ns: number of MCS samples
% GT_GP: quantile of response at target reliability (CDF) level (used
% as reliability constraint in RBDO)
% Sen_GT_GP: sensivities of quantile w.r.t. means of design variables
%
% FUNCTION DEFINITION
% RBDO_GP(): main function
% Costfun(): objective function & gradients w.r.t. means of design variables
% frelcon(): Define the probability constraints and gradients
% GP_RS(): GP method for reliability analysis under uncertainty
% SHOW(): show the optimization information for every design iteration
% FindResponse(): define performance functions

%% Updates for version22
% 1. 'Algorithm','sqp' -> 'Algorithm','interior-point', sigma = 0.6 sqp was getting stuck
% 2. GPmodel.noise_std = 1e-6 -> GPmodel.noise_std = 1e-5, minimize chol() errors
% 3. GPmodel.max_n_pt = 100; -> Make sure acquisiton functions are active
% 4. Add additional break statement to eliminate double points if
%    sigma_Cand value is close to GPmodel.noise_std
%       if max(accuracy(:,i))<accuracy_lim
%         break
%       elseif max(sigma_Cand(i_acqfunc_comp))<=1e-4
%          disp('Sigma Break')
%          break
%       end


%%

clc;
clear all;
close all;
clear fig_h
clear global

format short

global nc rt Iters nd stdx Cost GPmodel DDOIters x_train_vals fig_h mathfunc plottype h x_eval 

%================== Define Global or Local Window ========================%

%Specify Acquisition Function
%->'U'
%->'MMCE'
%->'EFF'
%->'EFF_miss_e'
%->'MCE'
%->'SEEDT'
%->'H'
%->'REIF'
%->'REIF2'
AcqFunc = 'EFF';

rng('default')
rs = 1; %EFF = rs=5 for F1 sigma = 0.6 Global and Local
        %EFF_miss_e = rs=1 for F1 sigma = 0.6 Global and Local

%Specify Window Size Options
%->'Local'
%->'Global'
window_size = 'Local';

%Specify Math Function
%-> F1
%-> F2
mathfunc = 'F1';

%Plot samples or estimate of limit state function
%->'samples'
%->'limits'
plottype = 'limits';

%Write files to disk?
%-> 'Yes'
%-> 'No'
outputfiles = 'No';


fig_h=figure(1);
% h1=1;

%==================    Define Optimization Parameters  ===================%
nd = 2;                             % Number of design variables
nc = 3;                             % Number of constraits
rt = 0.9987;                        % Target reliability
coeff_local_window = 1.3;           % Coefficient to expand a local window
d0 = [5, 5];                        % Initial design (initial mean of X)
accuracy_lim = 0.01;                %Accuracy limit
sigma = 0.6;                        %Standard deviation
stdx = [sigma,sigma];        % Standard deviation vector of X
lb = [0,0];
ub = [10,10];

%Specify values for output file naming
tempname1 = num2str(accuracy_lim);                   %Convert number to string
accuracy_name = [tempname1(1),'p',tempname1(3:4)];   %Name for output files
tempname2 = num2str(sigma);                      %Convert number to string
filesigmaname = [tempname2(1),'p',tempname2(3:end)]; %Name for output files

%Initilize variables
dp = d0;
Iters = 0;
DDOIters=0;

WD=cd;
subfold = [mathfunc,'_',window_size,'_',AcqFunc,'_sig_',filesigmaname,'_Acc_',accuracy_name];
filename = [mathfunc,'_',window_size,'_',AcqFunc,'_sig_',filesigmaname,'_Acc_',accuracy_name];

k1 = 10*rand(1000000,1);
k2 = 10*rand(1000000,1);
x_eval = [k1,k2];


DDO_options = optimoptions('fmincon',...
    'Algorithm','interior-point',...
    'StepTolerance',1e-6,...
    'ConstraintTolerance',1e-6,...
    'OptimalityTolerance',1e-6,...
    'MaxFunEvals',500);

options = optimoptions('fmincon','SpecifyConstraintGradient',true,...
    'SpecifyObjectiveGradient',true,...
    'Algorithm','interior-point',...
    'StepTolerance',1e-6,...
    'ConstraintTolerance',1e-6,...
    'OptimalityTolerance',1e-6,...
    'MaxFunEvals',500);

%================= Start Direct Design Optimization  =====================%
DDO = fmincon(@Costfun,d0,[],[],[],[],lb,ub,@frelconDDO,DDO_options);

dp = DDO;
GPmodel.variables.DDO = DDO;

%===================== Define DDO Constraints ============================%
    function [cc,cceq,GGC,GGCeq] = frelconDDO(d)
        cceq = []; GGC = []; GGCeq = [];

        c1 = FindResponse(d,1)';
        c2 = FindResponse(d,2)';
        c3 = FindResponse(d,3)';

        cc = [c1,c2,c3];

        dd = norm(d-dp);
        if  dd > 1d-6  || DDOIters == 0
            DDOIters = DDOIters + 1;
            SHOW(DDOIters,d,cc,GGC,@FindResponse,'DDO',[],GPmodel);
        end
        dp = d;

    end

%==================    Define GP Regression Parameters  ==================%
% rng('default')
% rs = 1;

ns_Cand = 10000;    % Number of MCS candidate points for sample selection
ns_Rel = 1000000;   % Number of MCS sample points for reliability analysis
p1 = DDO;           % Mean values of input random variables
p2 = stdx;          % Std values of input random variables
p3 = zeros(1,nd);   % Parameter "a" used for some non-normal distributions
p4 = zeros(1,nd);   % Parameter "b" used for some non-normal distributions

GPmodel.variables.ns_Cand = ns_Cand;
GPmodel.variables.ns_Rel = ns_Rel;
GPmodel.variables.lb = lb;
GPmodel.variables.ub = ub;
GPmodel.variables.accuracy_lim = accuracy_lim;

% ranges of hypercube = mean -/+ rgindex*std
GPmodel.variables.rgindex = norminv(rt,0,1)*coeff_local_window;
GPmodel.variables.para = [p1' p2' p3' p4'];
GPmodel.variables.range = [];
GPmodel.variables.dim = size(p1,2);   % Input dimension

% Distribution types available: normal; beta; uniform; exponential;
% gamma; weibull; lognormal; triangle
GPmodel.variables.type = ['normal.....';'normal.....'];
% GPmodel.variables.lb = lb;  %Lower bound limits
% GPmodel.variables.ub = ub;  %Upper bound limits
GPmodel.n_cons = nc;        % Number of rel. constraints (g functions)
GPmodel.target_Rel = rt;    % Target reliability
GPmodel.n_init = nd*4;         % Number of initial LHS sample points
GPmodel.max_n_pt = 100;      % Max number of sample points per new design
GPmodel.g_value = [];       % Initialize g values as empty vector
GPmodel.x_value = [];       % Initialize x values as empty vector
GPmodel.x_train = [];       % Add training variable for troubleshooting
GPmodel.g_train = [];       % Add training variable for troubleshooting
GPmodel.noise_std = 1e-5;   % Noise std (sigma_n)
GPmodel.candidates = [];
GPmodel.samples_Rel = [];
GPmodel.norm_candidates = [];
GPmodel.variables.graphcount=1;
GPmodel.random_seed = rs;       % Random seed for repeatability


d_all=[];
Rel_GP_all=[];
x_train_all=[];


%=======================    Start Optimization  ==========================%
[opt, fval, exitflag] = fmincon(@Costfun,DDO,[],[],[],[],lb,ub,@frelcon,options);

plot(opt(1),opt(2),'b.','MarkerSize',24); hold on;
xlim([0,10])
ylim([0,10])

if strcmp(outputfiles,'Yes')==1
    if not(isfolder([WD,'\',subfold]))
        mkdir([WD,'\',subfold]);
    end

    cd([WD,'\',subfold])
    print(filename,'-dsvg','-r600')
    cd(WD)
end


%=====================    Define Obj. Function  ==========================%
    function [f,g]= Costfun(d)

        if strcmp(mathfunc,'F2')==1
            f = -(((d(1)+d(2)-10).^2)./30)-(((d(1)-d(2)+10).^2)./120);
            g = [(-d(1)/12)-((d(2)-10)/20),(-d(2)/12)-(d(1)/20)+(5/6)];
            Cost=f;
        else
            f=d(1)+d(2);
            g=[1 1];
            Cost=f;
        end

    end

%====================  Define Constraints and Gradiants  =================%
    function [c,ceq,GC,GCeq] = frelcon(d)
        ceq = []; GCeq = [];

        % Update mean point to be the current design
        GPmodel.variables.para(:,1) = d';

        % Generate MCS points for new sample selection and reliability analysis
        GPmodel = GenerateSamples(GPmodel,'MCS',window_size);

        [GPmodel,PoF_GP,Sen_PoF_GP,Rel_GP] = GP_RS_fitrgp(GPmodel,@FindResponse,Iters,window_size,AcqFunc,fig_h,accuracy_lim,Rel_GP_all,WD,subfold,filename,outputfiles,plottype);
        c = PoF_GP - (1 - rt);
        GC = Sen_PoF_GP;

        Rel_GP_all = [Rel_GP_all;Rel_GP];
        d_all = [d_all;d];
        x_train_all = [x_train_all;size(GPmodel.x_value,2)];
        x_train_vals_all = GPmodel.x_value';
        x_train_vals = GPmodel.x_train;

        %Write output files for future analysis
        if strcmp(outputfiles,'Yes')==1

            if not(isfolder([WD,'\',subfold]))
                mkdir([WD,'\',subfold]);
            end

            cd([WD,'\',subfold])

         end


        dd = norm(d-dp);
        if  dd > 1d-6  || Iters == 0
            Iters = Iters + 1;
            SHOW(Iters,d,c,GC,@FindResponse,'RBDO',Rel_GP,GPmodel);
        end
        dp = d;
    end

%===================== Define Performance Functions ======================%
    function Response = FindResponse(x,cid)
        if isvector(x)
            if cid == 1
                Response = 1-((((x(1)).^2).*x(2))/20);

            elseif cid == 2
                if strcmp(mathfunc,'F2')==1
                    Y = 0.9063*x(1)+0.4226*x(2);
                    Z = 0.4226*x(1)-0.9063*x(2);
                    Response = -1+(Y-6).^2+(Y-6).^3-0.6*(Y-6).^4+Z;
                else
                    Response = 1-(((x(1)+x(2)-5).^2)/30)-(((x(1)-x(2)-12).^2)/120);
                end

            elseif cid == 3
                Response = 1-(80./((x(1).^2+8*x(2)+5)));
            end
        else
            if cid == 1
                Response = 1-((((x(1,:)).^2).*x(2,:))/20);

            elseif cid == 2
                if strcmp(mathfunc,'F2')==1
                    Y = 0.9063*x(1,:)+0.4226*x(2,:);
                    Z = 0.4226*x(1,:)-0.9063*x(2,:);
                    Response = -1+(Y-6).^2+(Y-6).^3-0.6*(Y-6).^4+Z;
                else
                    Response = 1-(((x(1,:)+x(2,:)-5).^2)/30)-(((x(1,:)-x(2,:)-12).^2)/120);
                end

            elseif cid == 3
                Response = 1-(80./(x(1,:).^2+8.*x(2,:)+5));
            end
        end
    end

%===================== Display Iteration Information================%
    function  SHOW(Iters,d,c,GC,gfun,flag,Rel_GP,model)
        fprintf(1,'\n********** Iter.%d ***********\n' ,Iters);
        disp(['Des.: ' sprintf('%6.6f  ',d)]);
        disp(['Obj.: ' sprintf('%6.6f',Cost)]);
        disp(['Cons.: ' sprintf('%6.6f  ',c)]);
        disp(['Rel.: ' sprintf('%6.9f  ',Rel_GP)]);

        

        if isequal(flag,'RBDO')
            disp(['Train Points: ' sprintf('%6.0f  ',size(model.x_value,2))]);
            disp(['Acquistion Function: ',AcqFunc])
            disp(['Accuracy: ' sprintf('%6.6f  ',model.variables.accuracy_lim)]);
            for k = 1:nd
                if k ==1
                    disp(['Sens.: ' sprintf('%6.6f  ',GC(k,:))]);
                else
                    disp(['           ' sprintf('%6.6f  ',GC(k,:))]);

                end
            end

            if strcmp(plottype,'limits')==1

                for j = 1:model.n_cons
                    % [mu_G_Rel0(:,j), sigma_Cand] = predict(model.gprMdl{j},x_eval);
                    [mu_G_RelCons, sigma_Cand] = predict(model.gprMdl{j},x_eval);
                    ConstraintIndex = [];
                    for j1 = 1:length(mu_G_RelCons)
                        if abs(mu_G_RelCons(j1)) < 0.01
                            ConstraintIndex = [ConstraintIndex;j1];
                        end
                    end

                    if isempty(ConstraintIndex)==1
                        ConstraintIndex = [1];
                    end

                    mu_G_Rel0.Constraint(j).Data = [k1(ConstraintIndex),k2(ConstraintIndex)];
                end


                colors = {'#76448A',...%'#CB4335',...
                    '#A7D2EF',... %#85C1E9
                    '#1F618D',...
                    '#82E0AA',...
                    '#1E8449',...
                    '#95A5A6',...
                    '#A569BD ',...
                    '#389AAE',... %#0868ac
                    '#717D7E'...
                    };


                figure(1)
                for j = 1:model.n_cons
                    if Iters==1
                        h(j).plothandle=[];

                    end
                    delete(h(j).plothandle)
                    figure(1)
                    h(j).plothandle = plot(mu_G_Rel0.Constraint(j).Data(:,1),mu_G_Rel0.Constraint(j).Data(:,2),'.','Color',colors{2*j+1});
                    h100 = plot(d(1),d(2),'b.','MarkerSize',24);
                    refresh(figure(1))
                    delete(h100);
                end
            end

        end

        fprintf('\n\n')

        if Iters ~= 100
            % To visualize the result, first create a grid of evenly spaced
            % points in two-dimensional space.
            x1 = -1:.1:10;
            x2 = -1:.1:10;
            [X1,X2] = meshgrid(x1,x2);
            X = [X1(:) X2(:)]';

            % Then, evaluate the performance functions at the grid points.
            if strcmp(mathfunc,'F1')==1
                c = X(1,:)+X(2,:);
            else
                c = -(((X(1,:)+X(2,:)-10).^2)/30)-(((X(1,:)-X(2,:)+10).^2)/120);
            end

            c = reshape(c,length(x2),length(x1));
            g1 = gfun(X,1);
            g1 = reshape(g1,length(x2),length(x1));
            g2 = gfun(X,2);
            g2 = reshape(g2,length(x2),length(x1));
            g3 = gfun(X,3);
            g3 = reshape(g3,length(x2),length(x1));

            % Finally, create a contour plot of the multivariate normal distribution
            % that includes the unit square.

            if Iters==1
                fig_h=figure(1);
                set(fig_h,'Name','Results')
                set(fig_h,'defaultAxesFontSize',12)
                set(fig_h,'defaultAxesFontName','Times')
                set(fig_h,'Position', [10 10 400 400])

                if strcmp(mathfunc,'F1')==1
                    contour(x1,x2,c,[1 3 5 7 9 11 13 15 17],'--',...
                        'Color',[0.9 0.9 0.9],'LineWidth',0.75); hold on;
                else
                    contour(x1,x2,c,[-5 -4.5 -4 -3.5 -3 -2.5 -2 -1.5 -1 -0.5 0 0.5 1],'--',...
                        'Color',[0.9 0.9 0.9],'LineWidth',0.75); hold on;
                end

                contour(x1,x2,g1,[0 0],'k-',...
                    'LineWidth',1.5); hold on;
                xa = 2; ya = 6;
                text(xa, ya, '{\itG}_{1} = 0','FontName','Times','FontSize',12); hold on;

                contour(x1,x2,g2,[0 0],'k-',...
                    'LineWidth',1.5); hold on;
                xa = 4.5; ya = 3;
                text(xa, ya, '{\itG}_{2} = 0','FontName','Times','FontSize',12); hold on;

                contour(x1,x2,g3,[0 0],'k-',...
                    'LineWidth',1.5); hold on;
                xa = 5; ya = 6.6;
                text(xa, ya, '{\itG}_{3} = 0','FontName','Times','FontSize',12); hold on;



                %                 plot(d(1),d(2),'b.','MarkerSize',24); hold on;

                xlabel('{\itx}_{1}','FontName','Times','FontSize',12)
                ylabel('{\itx}_{2}','FontName','Times','FontSize',12)
                xticks(linspace(0, 10, 11))
                yticks(linspace(0, 10, 11))
            else
                contour(x1,x2,g1,[0 0],'k-',...
                    'LineWidth',1.5); hold on;

                contour(x1,x2,g2,[0 0],'k-',...
                    'LineWidth',1.5); hold on;

                contour(x1,x2,g3,[0 0],'k-',...
                    'LineWidth',1.5); hold on;

                %                 plot(d(1),d(2),'k.','MarkerSize',24); hold on;

            end




        end

    end
%===============================================================%
end

