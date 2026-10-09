%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Make_Figure_10.m
%
% Generates Figure 10: Subsample Stability of Elasticity Estimates
%
%   Figure_10a.png - Leave-out window = 3 quarters
%   Figure_10b.png - Leave-out window = 19 quarters
%
% For each observation t, a window of (step+1) consecutive observations
% starting at t is excluded from the sample. The SP-IV elasticity of TFP
% with respect to government R&D capital is re-estimated on the remaining
% data. The figure plots the rolling subsample estimates with 95% Wald
% and weak-IV robust (KLM) confidence intervals, compared to the
% full-sample SP-IV estimate (dashed horizontal line).
%
% Specification: ln_tfp_util_adj2 on ln_rK_GRD using z_allNDEF
% Full-sample reference: same but using z_NDEF (baseline exogenous shocks)
%
% Source: DATAQ timetable (must be in workspace)
% Requires: btrace.m (in auxiliary_files/), ciplot.m, tightfig.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
warning off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1. SETTINGS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

varsy     = 'ln_tfp_util_adj2';
varsY     = {'ln_rK_GRD'};
Lcontrols = {'ln_cum_rALL_NDEF', 'ln_util', 'ln_tfp_util', ...
             'ln_rK_PRD_fernald', 'R_EW_COMB_TECH', ...
             'ln_rK_GRD', 'Military_news_scaled'};
horizons  = 3:4:59;
p         = 4;
clevel    = 0.95;

K  = 1;
H  = length(horizons);
Nz = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2. FULL-SAMPLE SP-IV ESTIMATE (reference line)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Baseline specification with exogenous ND shocks (z_NDEF)
bb = get_spiv_estimate(varsy, varsY, {'z_NDEF'}, [{varsy} Lcontrols], ...
                        horizons, p, DATAQ);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 3. DATA SETUP FOR LOO ANALYSIS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% LOO analysis uses all ND shocks (z_allNDEF)
all_Lcontrols = [{varsy} Lcontrols];
sel_hors_all  = 0:max(horizons);

y  = DATAQ(:, {varsy}).Variables';
Y  = DATAQ(:, varsY).Variables';
XL = lagmatrix(DATAQ(:, all_Lcontrols).Variables, 1:p)';
Z  = DATAQ(:, {'z_allNDEF'}).Variables';

IH = eye(H);
R  = kron(eye(K), IH(:));

% Multi-horizon outcomes: 4-quarter moving average of levels
yHt = (lagmatrix(y',-sel_hors_all)' + lagmatrix(y',-sel_hors_all+1)' + ...
       lagmatrix(y',-sel_hors_all+2)' + lagmatrix(y',-sel_hors_all+3)') / 4;
yH  = yHt(1+horizons, :);

YH = [];
for k = 1:K
    YHk = (lagmatrix(Y(k,:)',-sel_hors_all)' + lagmatrix(Y(k,:)',-sel_hors_all+1)' + ...
           lagmatrix(Y(k,:)',-sel_hors_all+2)' + lagmatrix(Y(k,:)',-sel_hors_all+3)') / 4;
    YH = [YH; YHk(1+horizons, :)];
end

% Drop NaN observations
sel = ~isnan(sum([yH; YH; XL; Z]));
yH = yH(:,sel);
YH = YH(:,sel);
XL = XL(:,sel);
Z  = Z(:,sel);
dates = DATAQ.Time(sel);

TT = size(yH, 2);
X  = [XL; ones(1,TT)];
Nx = size(X, 1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 4. LEAVE-OUT ANALYSIS AND FIGURES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

steps      = [3 19];
fig_labels = {'a', 'b'};

for s = 1:length(steps)
    step = steps(s);
    fprintf('Figure 10%s: step = %d\n', fig_labels{s}, step);

    [b_loo, cilw, ciuw, cil, ciu] = run_loo(yH, YH, Z, X, ...
        Nx, H, K, Nz, R, step, clevel, TT);

    % --- Plot ---
    f = figure;
    hold on
    pp = plot(dates, b_loo, 'Linewidth', 2);
    ciw = plot(dates, [ciuw cilw], 'Linewidth', 1, ...
        'Color', [0 0.4470 0.7410], 'LineStyle', '--');
    cis = ciplot(cil, ciu, dates, [0.5 0.5 0.5], 0.2);
    cis.LineStyle = 'none';
    c = yline(bb, '--');
    ylim([-0.1 0.35]);
    yline(0);
    l = legend([c; pp; ciw(1); cis], ...
        {'Full-Sample SP-IV Estimate', 'Subsample SP-IV Estimate', ...
         '95\% Wald CI', '95\% Weak-IV Robust CI'});
    set(l, 'Location', 'Best', 'Interpreter', 'Latex', 'Box', 'Off')
    xaxisproperties = get(gca, 'XAxis');
    xaxisproperties.TickLabelInterpreter = 'latex';
    yaxisproperties = get(gca, 'YAxis');
    yaxisproperties.TickLabelInterpreter = 'latex';
    tightfig;
    exportgraphics(gcf, sprintf('figures/Figure_10%s.png', fig_labels{s}), ...
        'Resolution', 300);
end

disp('Figure 10 generated successfully.');
warning on

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% LOCAL FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function b = get_spiv_estimate(varsy, varsY, varsZ, all_Lcontrols, ...
                                sel_hors, p, DATA)
% Compute full-sample SP-IV point estimate (K=1, 4Q MA of levels).

    K  = length(varsY);
    H  = length(sel_hors);
    Nz = length(varsZ);
    sel_hors_all = 0:max(sel_hors);

    y  = DATA(:, {varsy}).Variables';
    Y  = DATA(:, varsY).Variables';
    XL = lagmatrix(DATA(:, all_Lcontrols).Variables, 1:p)';
    Z  = DATA(:, varsZ).Variables';

    yHt = (lagmatrix(y',-sel_hors_all)' + lagmatrix(y',-sel_hors_all+1)' + ...
           lagmatrix(y',-sel_hors_all+2)' + lagmatrix(y',-sel_hors_all+3)') / 4;
    yH  = yHt(1+sel_hors, :);
    YH = [];
    for k = 1:K
        YHk = (lagmatrix(Y(k,:)',-sel_hors_all)' + lagmatrix(Y(k,:)',-sel_hors_all+1)' + ...
               lagmatrix(Y(k,:)',-sel_hors_all+2)' + lagmatrix(Y(k,:)',-sel_hors_all+3)') / 4;
        YH = [YH; YHk(1+sel_hors, :)];
    end

    sel = ~isnan(sum([yH; YH; XL; Z]));
    yH=yH(:,sel); YH=YH(:,sel); XL=XL(:,sel); Z=Z(:,sel);
    T = size(yH, 2);

    X  = [XL; ones(1,T)];
    yHo = yH - (yH/X)*X;
    YHo = YH - (YH/X)*X;
    Zo  = Z  - (Z/X)*X;

    Sig_Zo = Zo*Zo'/T;
    irfsy  = (yHo/Zo) * Sig_Zo^0.5;
    irfsY  = (YHo/Zo) * Sig_Zo^0.5;

    bThetay = reshape(irfsy, 1, H*Nz)';
    bThetaY = [];
    for k = 1:K
        bThetaY(:,k) = reshape(irfsY(1+(k-1)*H:k*H, :), H*Nz, 1);
    end
    b = bThetay' * bThetaY / (bThetaY' * bThetaY);
end


function [b_loo, cilw, ciuw, cil, ciu] = run_loo(yH, YH, Z, X, ...
    Nx, H, K, Nz, R, step, clevel, TT)
% Leave-out SP-IV estimation: for each t, exclude observations t:t+step,
% re-estimate elasticity, and compute Wald + KLM confidence intervals.

    b_loo = NaN(TT, 1);
    cilw  = NaN(TT, 1);
    ciuw  = NaN(TT, 1);
    cil   = NaN(TT, 1);
    ciu   = NaN(TT, 1);

    Ngrid = 501;
    bgrid = linspace(-0.25, 1, Ngrid);

    for t = 1:TT
        sel_loo = ones(TT, 1);
        sel_loo(t:min(t+step, TT)) = 0;

        yH_loo  = yH(:, sel_loo==1);
        YH_loo  = YH(:, sel_loo==1);
        Z_loo   = Z(:, sel_loo==1);
        X_loo   = X(:, sel_loo==1);
        yHo_loo = yH_loo - (yH_loo/X_loo)*X_loo;
        YHo_loo = YH_loo - (YH_loo/X_loo)*X_loo;
        Zo_loo  = Z_loo  - (Z_loo/X_loo)*X_loo;

        T = size(yH_loo, 2);

        % First-stage projections
        PYHo_loo   = (YHo_loo/Zo_loo)*Zo_loo;
        Sig_Zo_loo = Zo_loo*Zo_loo'/T;
        irfsy_loo  = (yHo_loo/Zo_loo) * Sig_Zo_loo^0.5;
        irfsY_loo  = (YHo_loo/Zo_loo) * Sig_Zo_loo^0.5;

        % Second stage: SP-IV estimate
        bThetay_loo = reshape(irfsy_loo, 1, H*Nz)';
        bThetaY_loo = [];
        for k = 1:K
            bThetaY_loo(:,k) = reshape(irfsY_loo(1+(k-1)*H:k*H, :), H*Nz, 1);
        end
        b_loo(t) = bThetay_loo' * bThetaY_loo / (bThetaY_loo' * bThetaY_loo);

        % Wald confidence interval
        uHo_loo    = yHo_loo - kron(b_loo(t), eye(H)) * YHo_loo;
        SiguHo_loo = uHo_loo*uHo_loo' / (T-K-Nx);
        Sig_beta   = (R'*kron(PYHo_loo*PYHo_loo',eye(H))*R) \ ...
                     R'*kron(PYHo_loo*PYHo_loo',SiguHo_loo)*R / ...
                     (R'*kron(PYHo_loo*PYHo_loo',eye(H))*R);
        ciuw(t) = b_loo(t) + tinv(1-(1-clevel)/2, T-K-Nx)*sqrt(Sig_beta);
        cilw(t) = b_loo(t) - tinv(1-(1-clevel)/2, T-K-Nx)*sqrt(Sig_beta);

        % KLM confidence interval (1D grid search)
        S_KLM   = zeros(Ngrid, 1);
        OBJ_KLM = zeros(Ngrid, 1);
        for c = 1:Ngrid
            uHo_c = yHo_loo - kron(bgrid(c), eye(H)) * YHo_loo;
            S_KLM(c)   = GetKstat(uHo_c, Zo_loo, YHo_loo, Nx, H, K);
            OBJ_KLM(c) = GetARObj(uHo_c, Zo_loo, YHo_loo);
        end

        % Walk outward from minimum to find CI boundaries
        start = find(OBJ_KLM == min(OBJ_KLM));

        j = 1;  flag = 0;
        while flag == 0
            in = start + j;
            if in > length(bgrid)
                flag = 1;
            elseif S_KLM(in) > chi2inv(clevel, K)
                flag = 1;
            end
            ciu(t) = bgrid(in-1);
            j = j + 1;
        end

        j = 1;  flag = 0;
        while flag == 0
            in = start - j;
            if in < 1
                flag = 1;
            elseif S_KLM(in) > chi2inv(clevel, K)
                flag = 1;
            end
            cil(t) = bgrid(in+1);
            j = j + 1;
        end
    end
end


function AR_OBJ = GetARObj(uH, Z, YH)
    [~,T] = size(Z);
    PuH = (uH/Z)*Z;
    MuH = uH - PuH;
    AR_OBJ = T * trace(PuH*PuH' / (MuH*MuH'));
end


function Kstat = GetKstat(uH, Z, YH, Nx, H, K)
    [Nz,T] = size(Z);
    IH = eye(H);
    R  = kron(eye(K), IH(:));
    PYH = (YH/Z)*Z;
    vH  = YH - PYH;
    PuH = (uH/Z)*Z;
    MuH = uH - PuH;
    MvH = vH - (vH/Z)*Z;
    V   = MuH*MuH';
    iV  = eye(H)/V;
    tYhat = PYH - MvH/MuH*PuH;
    Sigg  = R'*kron(tYhat*tYhat', iV*uH*uH'*iV)*R / (T-Nz-Nx);
    iSigg = eye(K)/Sigg;
    Kstat = btrace(iV*uH*tYhat', H) * iSigg * btrace(iV*uH*tYhat', H)';
end
