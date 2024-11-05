function model = GenerateSamples(model,method,window_size)

% Generate MCS points for new sample selection and reliability analysis
rgindex = model.variables.rgindex;
para = model.variables.para;
type = model.variables.type;
nv = size(para,1);
a = zeros(1,nv); b = zeros(1,nv);
p = zeros(1,nv); q = zeros(1,nv); r = zeros(1,nv);
lb = model.variables.lb';
ub = model.variables.ub';


for k = 1:nv
    if strcmp('normal.....', type(k,:))
            u(k) = para(k,1);
            s(k) = para(k,2);
            range(k,:) = [(u(k)-rgindex*s(k))',(u(k)+rgindex*s(k))'];
    elseif strcmp('beta.......', type(k,:))
        u(k) = para(k,1);
        s(k) = para(k,2);
        a(k) = para(k,3);
        b(k) = para(k,4);
        r(k) = b(k)-a(k);
        p(k) = (-3*u(k)*a(k)^2+3*u(k)^2*a(k)+s(k)^2*a(k)-s(k)^2*u(k)...
            +r(k)*u(k)^2+a(k)^3-u(k)^3+r(k)*a(k)^2-2*r(k)*u(k)*a(k))...
            /s(k)^2/r(k);
        q(k) = (-r(k)+u(k)-a(k))*(u(k)^2-2*u(k)*a(k)-r(k)*u(k)+a(k)^2....
            +r(k)*a(k)+s(k)^2)/s(k)^2/r(k);
        range(k,:) = [a(k),b(k)];
    elseif strcmp('uniform....', type(k,:))
        u(k) = para(k,1);
        s(k) = para(k,2);
        a(k) = para(k,1);  % u(k)-1/2*sqrt(12)*s(k);
        b(k) = para(k,2);  % u(k)+1/2*sqrt(12)*s(k);
        range(k,:) = [a(k),b(k)];
    elseif strcmp('triangle...', type(k,:))
        a(k) =  para(k,1);
        b(k) =  para(k,3);
        p(k) =  para(k,2);
        range(k,:) = [a(k),b(k)];
    elseif strcmp('lognormal..', type(k,:))
        u(k) = para(k,1);
        s(k) = para(k,2);
        q(k) = sqrt(log(1+(s(k)/u(k))^2));
        p(k) = log(u(k))-0.5*q(k)^2;
    elseif strcmp('exponential', type(k,:))
        u(k) = para(k,1);
        s(k) = para(k,1);
    elseif strcmp('weibull....', type(k,:))
        p(k) = para(k,1);
        q(k) = para(k,2);
        a(k) = para(k,3);
        u(k) = p(k)*gamma(1+1/q(k))+a(k);
        s(k) = sqrt(p(k)^2*(gamma(1+2/q(k))-(gamma(1+1/q(k)))^2));
        range(k,:) = [max((u(k)-rgindex*s(k))',0),(u(k)+rgindex*s(k))'];
    else
        error('Distribution types are not defined correctly!\n');
    end
end
model.variables.range = range;

%======    Generate Uniform Random Seeds for Reliability Analysis  =======%
rs = model.random_seed;
if ~isempty(rs)
    rng(rs)         % Control random number generation for repeatability
end

ns_Cand = model.variables.ns_Cand;
ns_Rel = model.variables.ns_Rel;
for k = 1:nv
    if strcmp('normal.....', type(k,:))
        if isequal('Local',window_size)
            %             xs_Cand(k,:) = normrnd(u(k),s(k),1,ns_Cand); %Original code
            init_Cand(k,:) = lhsdesign(ns_Cand,1).'; %Number of initial LHS points
            xs_Cand(k,:) = range(k,1) + (range(k,2) - range(k,1))*init_Cand(k,:);
            xs_Rel(k,:) = normrnd(u(k),s(k),1,ns_Rel);
        else
            xs_Cand(k,:) = rand(ns_Cand,1)*(range(k,2)-range(k,1));
            xs_Rel(k,:) = normrnd(u(k),s(k),1,ns_Rel);
        end
    elseif strcmp('beta.......', type(k,:))
        xs_Cand(k,:) = betarnd(p(k),q(k),1,ns_Cand)*r(k)+a(k);
        xs_Rel(k,:) = betarnd(p(k),q(k),1,ns_Rel)*r(k)+a(k);
    elseif strcmp('uniform....', type(k,:))
        xs_Cand(k,:) = unifrnd(a(k),b(k),1,ns_Cand);
        xs_Rel(k,:) = unifrnd(a(k),b(k),1,ns_Rel);
    elseif strcmp('exponential', type(k,:))
        xs_Cand(k,:) = exprnd(u(k),1,ns_Cand);
        xs_Rel(k,:) = exprnd(u(k),1,ns_Rel);
    elseif strcmp('weibull....', type(k))
        xs_Cand(k,:) = wblrnd(p(k),q(k),1,ns_Cand)+a(k);
        xs_Rel(k,:) = wblrnd(p(k),q(k),1,ns_Rel)+a(k);
    elseif strcmp('triangle...', type(k,:))
        temp = unifrnd(0,1,1,ns_Cand);
        for k1 = 1:ns_Cand
            if mod(k1,10000) == 0
                k1;
            end
            xs_Cand(k,k1) = trirnd(a(k),p(k),b(k),temp(k1));
        end
        temp = unifrnd(0,1,1,ns_Rel);
        for k1 = 1:ns_Rel
            if mod(k1,10000) == 0
                k1;
            end
            xs_Rel(k,k1) = trirnd(a(k),p(k),b(k),temp(k1));
        end
    elseif strcmp('lognormal', type(k,:))
        xs_Cand(k,:) = lognrnd(p(k),q(k),1,ns_Cand);
        xs_Rel(k,:) = lognrnd(p(k),q(k),1,ns_Rel);
    end
end

model.candidates = xs_Cand;

% model.candidatesplot = xs_Cand_plot;
for k = 1:nv
    norm_Cand(k,:) = (xs_Cand(k,:) - range(k,1))/(range(k,2)-range(k,1));
end
model.norm_candidates = norm_Cand;

model.samples_Rel = xs_Rel;

if strcmp('DATA', method)
    load('xs_Cand','xs_Cand');
    model.candidates=xs_Cand;
end

end
