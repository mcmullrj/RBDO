function [model, PoF_GP, Sen_PoF_GP, Rel_GP] = GP_RS_fitrgp(model,gfun,Iters,window_size,AcqFunc,fig_h,accuracy_lim,Rel_GP_all,WD,subfold,filename,Outputfiles,plottype)

% GP_RS_v1 created by Chao Hu on March 27, 2022
% GT_GP: quantile of response at target reliability (CDF) level (used
% as reliability constraint in RBDO)
% Sen_GT_GP: sensivities of quantile w.r.t. means of design variables


nv = model.variables.dim;   % # of variables
mu_x = model.variables.para(:,1);
sigma_x = model.variables.para(:,2);
range = model.variables.range; %ranges of hypercube = mean -/+ rgindex*std
nc = model.n_cons;          % # of constraints
rt = model.target_Rel;      % reliability target
lb = model.variables.lb';
ub = model.variables.ub';
globalbounds = [lb,ub];
sigma_n = model.noise_std;


% Identify initial LHS points
rs = model.random_seed;     % random number seed for repeatability
rng(rs); %random number generator

if isempty(model.x_value)   % No sampling done (initial sampling)


    % LHS sampling over unit hypercube, [0,1]^nv
    init_pt = lhsdesign(model.n_init,nv);
    init_pt = init_pt';

    % Transformation to original space
    
   if strcmp(window_size,'Local')==1
    for k = 1:nv
        init_pt(k,:) = range(k,1) + (range(k,2) - range(k,1))*init_pt(k,:);
    end
   else
       for k = 1:nv
        init_pt(k,:) = globalbounds(k,1) + (globalbounds(k,2) - globalbounds(k,1))*init_pt(k,:);
       end
   end

    % Grab G function values at LHS points
    for i = 1:size(init_pt,2)%model.n_init
        for j = 1:nc
            init_g(j,i) = gfun(init_pt(:,i),j);
        end
    end
    model.x_value = init_pt;
    model.g_value = init_g;

    x_train = model.x_value;
    g_train = model.g_value;

        % set(0,'CurrentFigure',fig_h)
        fig_h;
        plot(x_train(1,:),x_train(2,:),'kx','MarkerSize',10,'LineWidth',1.5)
        hold on
        xlim([0,10])
        ylim([0,10])

    model.x_train = x_train;


    disp('fitrgp #1')
    for j = 1:nc
        model.gprMdl{j} = fitrgp(x_train',g_train(j,:)','Basis','none',...
            'KernelFunction','squaredexponential',...
            'Sigma',sigma_n+1e-6,'Standardize',true,'ConstantSigma',true,...
            'SigmaLowerBound',sigma_n,...
            'FitMethod','exact','PredictMethod','exact');
    end


 
    if strcmp(Outputfiles,'Yes')==1
        if not(isfolder([WD,'\',subfold]))
            mkdir([WD,'\',subfold]);
        end
        cd([WD,'\',subfold])
        format shortg
        tmck = fix(clock);
        clockstring = [num2str(tmck(1)),num2str(tmck(2)),num2str(tmck(3)),num2str(tmck(4)),num2str(tmck(5)),num2str(tmck(6))];
        
        model_save = model;
        model_save.candidates = []; %Remove to save file size
        model_save.samples_Rel = []; %Remove to save file size
        model_save.norm_candidates = []; %Remove to save file size

        save([filename,'_',clockstring,'.mat'],'x_train','model_save')
        cd(WD)
    end

else                        % Some samples exist
    n_x_value = size(model.x_value,2);  % Total number of x points

    % Step 1: Identify within-range x points for reuse
    LB_range = repmat(range(:,1),1,n_x_value);  % Define lower bound matrix
    UB_range = repmat(range(:,2),1,n_x_value);  % Define upper bound matrix

    sum_logical = sum(model.x_value - LB_range >= 0,1) +...
        sum(UB_range - model.x_value >= 0,1);
    ind_within_range = find(sum_logical == 2*nv);

    % Step 2: Generate LHS samples for a new design if number of existing
    % samples falling within range is less than model.n_init
    if length(ind_within_range) < model.n_init
        n_init_add = model.n_init - length(ind_within_range);

        % LHS sampling over unit hypercube, [0,1]^nv
        init_pt_add = lhsdesign(model.n_init*2,nv);
        init_pt_add =...  % Skip model.n_init LHS points used earlier
            init_pt_add((model.n_init+1) : (model.n_init+n_init_add),:)';

        % Transformation to original space
        for k = 1:nv
            init_pt_add(k,:) = range(k,1) +...
                (range(k,2) - range(k,1))*init_pt_add(k,:);
        end

        % Grab G function values at LHS points
        for i = 1:n_init_add
            for j = 1:nc
                init_g_add(j,i) = gfun(init_pt_add(:,i),j);
            end
        end
        model.x_value = [model.x_value,init_pt_add];
        model.g_value = [model.g_value,init_g_add];

        x_train = [model.x_value(:,ind_within_range),init_pt_add];
        g_train = [model.g_value(:,ind_within_range),init_g_add];

        model.x_train = x_train;

        disp('fitrgp #2')
        for j = 1:nc
            model.gprMdl{j} = fitrgp(x_train',g_train(j,:)','Basis','none',...
                'KernelFunction','squaredexponential',...
                'Sigma',sigma_n+1e-5,'Standardize',true,'ConstantSigma',true,...
            'SigmaLowerBound',sigma_n,...
            'FitMethod','exact','PredictMethod','exact');
        end


    else
        % number of existing samples falling within range is larger than or
        % equal to model.n_init
        x_train = [model.x_value(:,ind_within_range)];
        g_train = [model.g_value(:,ind_within_range)];

    end
end


% Start sequential sampling using the MEU acquisition function
xs_Cand = model.candidates;     % Extract MCS points for sample selection
cov_x = diag(sigma_x.^2);
pdfx = mvnpdf(xs_Cand',mu_x',cov_x);

if strcmp(plottype,'samples')==1
    figure(1)
    scatter(xs_Cand(1,:),xs_Cand(2,:),'filled','MarkerFaceAlpha',.02,'MarkerEdgeAlpha',.02)
    hold on
    ylim([0,10])
end

if strcmp(window_size,'Global')==1
    model.max_n_pt = model.max_n_pt+1;
end


for i = 1:(model.max_n_pt - size(x_train,2))


    for j = 1:nc
        [mu_Cand, sigma_Cand] = predict(model.gprMdl{j},xs_Cand');
        sigma_Cand2=sigma_Cand;

        upperbound = mu_Cand + 3*sigma_Cand2;
        lowerbound = mu_Cand - 3*sigma_Cand2;
        accuracy(j,i) = size(find(lowerbound.*upperbound <= 0),1)/size(mu_Cand,1);

        if isequal('U',AcqFunc) %AKMCS
            acqfunc(:,j) = abs(mu_Cand)./sigma_Cand2;

        elseif isequal('MMCE',AcqFunc) %AKMCS_pdf
            CL2 = normcdf(abs(mu_Cand)./sigma_Cand2);
            acqfunc(:,j) = (1-CL2).*pdfx;

        elseif isequal('EFF',AcqFunc)
            es = 2*sigma_Cand2;
            EFF_A = 2*normcdf(-mu_Cand./sigma_Cand2);
            EFF_B = normcdf((-es-mu_Cand)./sigma_Cand2);
            EFF_C = normcdf((es-mu_Cand)./sigma_Cand2);
            EFF_1 = (mu_Cand).*(EFF_A-EFF_B-EFF_C);
            EFF_D = 2*normpdf(-mu_Cand./sigma_Cand2);
            EFF_E = normpdf((-es-mu_Cand)./sigma_Cand2);
            EFF_F = normpdf((es-mu_Cand)./sigma_Cand2);
            EFF_2 = (sigma_Cand2).*(EFF_D-EFF_E-EFF_F);
            EFF_3 = es.*(EFF_C-EFF_B);
            acqfunc(:,j) = EFF_1-EFF_2+EFF_3;
%             disp('EFF')

        elseif isequal('EFF_miss_e',AcqFunc)
            es = 2*sigma_Cand2;
            EFF_A = 2*normcdf(-mu_Cand./sigma_Cand2);
            EFF_B = normcdf((-es-mu_Cand)./sigma_Cand2);
            EFF_C = normcdf((es-mu_Cand)./sigma_Cand2);
            EFF_1 = (mu_Cand).*(EFF_A-EFF_B-EFF_C);
            EFF_D = 2*normpdf(-mu_Cand./sigma_Cand2);
            EFF_E = normpdf((-es-mu_Cand)./sigma_Cand2);
            EFF_F = normpdf((es-mu_Cand)./sigma_Cand2);
            EFF_2 = (sigma_Cand2).*(EFF_D-EFF_E-EFF_F);
            EFF_3 = (EFF_C-EFF_B);
            acqfunc(:,j) = EFF_1-EFF_2+EFF_3;

        elseif isequal('MCE',AcqFunc) %CL

            CL = normcdf(abs(mu_Cand)./sigma_Cand2);
            acqfunc(:,j) = (1-CL).*pdfx.*sigma_Cand2;


        elseif isequal('SEEDT',AcqFunc) %MEU

            alpha = exp(accuracy(j,i));
            acqfunc(:,j) = (((alpha.^2.*sigma_Cand2)./sqrt(1+alpha.^2)).*...
                exp(-(mu_Cand.^2)./(2*sigma_Cand2.*sqrt(1+alpha^2)))).*pdfx;

        elseif isequal('H',AcqFunc) %H

            GminusU = (2*sigma_Cand2-mu_Cand);
            GplusU = (2*sigma_Cand2+mu_Cand);
            negGminusU = (-2*sigma_Cand2-mu_Cand);

            acqfunc(:,j) = abs(log(sqrt(2*pi).*sigma_Cand2+(1/2)).*...
                (normcdf(GminusU./sigma_Cand2)-normcdf(negGminusU./sigma_Cand2))-...
                ((GminusU/2).*normpdf(GminusU./sigma_Cand2)+(GplusU/2).*normpdf(negGminusU./sigma_Cand2)));

        elseif isequal('REIF',AcqFunc) %REIF
            beta = mu_Cand./sigma_Cand2;
            w = 2;
            acqfunc(:,j)=mu_Cand.*(1-2.*normcdf(beta))+sigma_Cand2.*(w-sqrt(2/pi).*exp(-0.5.*beta.^2));


        elseif isequal('REIF2',AcqFunc) %REIF2
            beta = mu_Cand./sigma_Cand2;
            w = 2;
            acqfunc(:,j)=mu_Cand.*pdfx.*(1-2*normcdf(beta))+sigma_Cand2.*pdfx.*(w-sqrt(2/pi)*exp(-0.5.*beta.^2));

        elseif isequal('MPLF',AcqFunc)
            p=3;

            MPLF_ipos = find(mu_Cand>=0);
            MPLF_ineg = find(mu_Cand<0);
            acqfunc(MPLF_ipos,j)=sign(mu_Cand(MPLF_ipos)).*abs(mu_Cand(MPLF_ipos)).^p./sigma_Cand2(MPLF_ipos);
            acqfunc(MPLF_ineg,j)=sign(mu_Cand(MPLF_ineg)).*abs(mu_Cand(MPLF_ineg)).^p.*sigma_Cand2(MPLF_ineg);
        else
            error('Please Specify Acquistion Function from List')
        end
    end

    acqfunc = acqfunc';   % Every column contains acquisition function values

    if isequal('U',AcqFunc)
        acqfunc_composite = min(acqfunc,[],1);
        [~, i_acqfunc_comp] = min(acqfunc_composite);
    elseif isequal('MMCE',AcqFunc)
        acqfunc_composite = max(acqfunc,[],1); %Switched to maximum 5/6/2022
        [~, i_acqfunc_comp] = max(acqfunc_composite);
    elseif isequal('EFF',AcqFunc)
        acqfunc_composite = max(acqfunc,[],1);
        [~, i_acqfunc_comp] = max(acqfunc_composite);
    elseif isequal('MCE',AcqFunc)
        acqfunc_composite = max(acqfunc,[],1);
        [~, i_acqfunc_comp] = max(acqfunc_composite);
    elseif isequal('SEEDT',AcqFunc)
        acqfunc_composite = max(acqfunc,[],1);
        [~, i_acqfunc_comp] = max(acqfunc_composite);
    elseif isequal('EFF_miss_e',AcqFunc)
        acqfunc_composite = max(acqfunc,[],1);
        [~, i_acqfunc_comp] = max(acqfunc_composite);
    elseif isequal('H',AcqFunc)
        acqfunc_composite = max(acqfunc,[],1);
        [~, i_acqfunc_comp] = max(acqfunc_composite);
    elseif isequal('REIF',AcqFunc)
        acqfunc_composite = max(acqfunc,[],1);
        [~, i_acqfunc_comp] = max(acqfunc_composite,[],2);
    elseif isequal('REIF2',AcqFunc)
        acqfunc_composite = max(acqfunc,[],1);
        [~, i_acqfunc_comp] = max(acqfunc_composite);

    elseif isequal('MPLF',AcqFunc)
        acqfunc_composite = min(acqfunc,[],1);
        [~, i_acqfunc_comp] = min(acqfunc_composite);
    else
        error('Please Specify Acquistion Function from List')
    end

    clear acqfunc


    if max(accuracy(:,i))<accuracy_lim
        disp('Accuracy Break')
        maxaccuracy = max(accuracy(:,i));
        break
    elseif max(sigma_Cand(i_acqfunc_comp))<=1e-4
        disp('Sigma Break')
        break
    end

    x_star = xs_Cand(:,i_acqfunc_comp);
    sigma_star=sigma_Cand(i_acqfunc_comp);


             % set(0,'CurrentFigure',fig_h)
             fig_h;
             plot(x_star(1),x_star(2),'k^','MarkerSize',10,'LineWidth',1); hold on


    % Grab G function values at newly selected x point
    for jj = 1:nc
        g_star(jj,:) = gfun(x_star,jj);
    end


    % Update global training set by adding new x point and g values

    model.x_value = [model.x_value,x_star];
    model.g_value = [model.g_value,g_star];
    x_train = [x_train,x_star];
    g_train = [g_train,g_star];

    output=[x_train;g_train];

    model.x_train = x_train;


    % Update local kriging models in preparation for selecting next x point
    %     disp('fitrgp #3')
    for j = 1:nc
        model.gprMdl{j} = fitrgp(x_train',g_train(j,:)','Basis','none',...
            'KernelFunction','squaredexponential',...
            'Sigma',sigma_n+1e-5,'Standardize',true,'ConstantSigma',true,...
            'SigmaLowerBound',sigma_n,...
            'FitMethod','exact','PredictMethod','exact');
    end

end


% if model.max_n_pt - size(x_train,2) >= 1
%     accuracy;
% end



%===========  Conduct Reliability and Sensitivity Analysis  ==============%
xs_Rel = model.samples_Rel;     % Extract MCS points for rel. analysis
ns_Rel = model.variables.ns_Rel;
GT_GP = zeros(1,nc);
Sen_PoF_GP = zeros(nv,nc);
Sen_GUtoU_GP = zeros(nv,nc);
Sen_GStoU_GP = zeros(nv,nc);

%plot(x_train(1,:),x_train(2,:),'kx','MarkerSize',8,'LineWidth',1.5) %$$$


if strcmp(Outputfiles,'Yes')==1

    cd([WD,'\',subfold])
    format shortg
    tmck = fix(clock);
    clockstring = [num2str(tmck(1)),num2str(tmck(2)),num2str(tmck(3)),num2str(tmck(4)),num2str(tmck(5)),num2str(tmck(6))];
    
        model_save2 = model;
        model_save2.candidates = []; %Remove to save file size
        model_save2.samples_Rel = []; %Remove to save file size
        model_save2.norm_candidates = []; %Remove to save file size
    
    save([filename,'_',clockstring,'.mat'],'x_train','model_save2')
%     print([filename,'_',clockstring],'-dsvg','-r600') %Not printing out
%     figure to save hard disk space
    cd(WD)

end


for j = 1:nc

    % Reliability analysis using MCS points
    [mu_G_Rel, sigma_Cand] = predict(model.gprMdl{j},xs_Rel');
    mu_G_Rel = mu_G_Rel';
    GT_GP(j) = quantile(mu_G_Rel, rt);  % Quantile of g function at target
    % reliability (CDF) level rt
    Rel_GP(j) = length(find(mu_G_Rel <= 0))/ns_Rel;
    PoF_GP(j) = 1 - Rel_GP(j);

    %%%%% For calculating LSF approximation error (optional) %%%%
    Response_True = gfun(xs_Rel,j);
    GT_True(j) = quantile(Response_True, rt);
    % Rel_True = length(find(Response_True <= 0))/ns_Rel;

    epsilon = 0.05*sum(abs(Response_True))/size(Response_True,2);
    i_index = abs(epsilon) > abs(Response_True);
    LSF_error = sum(i_index.*abs(Response_True-mu_G_Rel))/sum(i_index);
    %%%%% For calculating LSF approximation error (optional) %%%%

    %     if PoF_GP >= 0 %0.0001

    % Sensitivity analysis using MCS points
    % Compute First-Order Sensitivity of Reliability using Score Function
    % Refer to "Stochastic sensitivity analysis by dimensional
    % decomposition and score functions", Probabilistic Engineering
    % Mechanics, Volume 24, Issue 3, July 2009, Pages 278-287
    for k = 1:nv
        % Step 1: compute dmu(G)/du = E[G*s], s: score function
        Sen_GUtoU_GP(k,j) = mean(mu_G_Rel.*((xs_Rel(k,:) - mu_x(k))/sigma_x(k)^2));
        Failed_Sample_Ind = mu_G_Rel > 0;
        Sen_PoFtoU_GP(k,j) = sum((xs_Rel(k,Failed_Sample_Ind) -...
            mu_x(k))/sigma_x(k)^2)/ns_Rel;

        % Step 2: compute dstd(G)/du = (E[G^2*s] - 2*E[G]*E[G*s])/(2*std(G)),
        % s: score function
        Sen_GStoU_GP(k,j) = mean(mu_G_Rel.^2.*(xs_Rel(k,:) - mu_x(k))/sigma_x(k)^2) ....
            - 2*mean(mu_G_Rel)*Sen_GUtoU_GP(k,j);
        Sen_GStoU_GP(k,j) = Sen_GStoU_GP(k,j)/2/std(mu_G_Rel);

        % Step 3: compute dGT/du =~ dmu(G)/du + 3*dstd(G)/du
        % (an approximation for rt = 0.9978, i.e., betat = 3)
        % Sen_PoF_GP(k,j) = Sen_GUtoU_GP(k,j) + 3*Sen_GStoU_GP(k,j);
        Sen_PoF_GP(k,j) = Sen_PoFtoU_GP(k,j);
    end

end


end

%==================  Conduct Sensitivity Analysis  =======================%

%
% %===============  Plot Probability Density Functions  ====================%
% % Plot PDF computed from univariate Response surface
% figure(1); axes('Fontsize',20);
% D = mu_G_Rel;
% [cn,xout] = hist(D,100);
% sss = sum(cn);
% unit=(xout(100)-xout(1))/99;
% for k = 1:100
%     cn(k)=cn(k)/sss/unit;
% end
% plot(xout,cn,'k-'); hold on;
% clear D cn xout;
%
% % Plot True PDF from direct MCS
% D = Response_True;
% [cn,xout] = hist(D,100);
% sss = sum(cn);
% unit=(xout(100)-xout(1))/99;
% for k = 1:100
%     cn(k)=cn(k)/sss/unit;
% end
% plot(xout,cn,'r*'); hold on;
% clear D cn xout;
% legend('RS-UDR','MCS');
% xlabel('{\itg}({\bfx})'); ylabel('Probability density');



%==================  Obtain Univariate Samples  ==========================%
function [output,input,gg] = UDR_sampling(gfun,mu,stdx,cid)
u_loc = [-3.0,-1.5,0,1.5,3.0]; %% Sample locations: [u+/-3.0s, u+/-1.5s, u]
nv = length(mu);                %% Dimension of the problem
m = length(u_loc);             %% Number of samples along each dimension
input = zeros(nv,m);
for k = 1:nv
    % Identify sample location
    input(k,:) = mu(k) + u_loc*stdx(k);

    % Get Response values
    xx = mu;
    for kk = 1:m
        xx(k) = input(k,kk);
        if isequal(k,1) && isequal(xx,mu)  %% Avoid re-evaluating mean value
            output(k,kk) = gfun(xx,cid);
            gg = output(k,kk);
        elseif ~isequal(k,1) && isequal(xx,mu)
            output(k,kk) = gg;
        else
            output(k,kk) = gfun(xx,cid);
        end
    end
end
end
