%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Make_Figure_E5.m
%
% Generates Appendix Figure E.5: Nondefense Government R&D, Estimates
% Assuming Different Production Function Elasticities to Business-Sector
% R&D Capital
%
% Panel (a): SP-IV production function elasticity (phi_hat) for five
%            private R&D elasticity multipliers: 1, 1.5, 2, 2.5, 3 x xi.
% Panel (b): SP-IV gross rate of return (rho_hat) for the same five
%            multipliers.
%
% Output:  figures/Figure_E5a.png  (elasticity bar chart)
%          figures/Figure_E5b.png  (gross rate of return bar chart)
%
% Source:  DATAQ timetable (must be in workspace)
% Requires: btrace.m, nancumsum.m, tightfig.m (auxiliary_files/)
%           knnsearch (Statistics and Machine Learning Toolbox)
%           fminunc   (Optimization Toolbox)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
warning off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1. SETTINGS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clevel   = [0.90 0.95 0.99];
horizons = 3:4:59;
p_lags   = 4;

Lcontrols_ND = {'ln_cum_rALL_NDEF', 'ln_util', 'ln_tfp_util', ...
                'ln_rK_PRD_fernald', 'R_EW_COMB_TECH', ...
                'ln_rK_GRD', 'Military_news_scaled'};

grid = [1 1.5 2 2.5 3] - 1;
N_priv = length(grid);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2. CONSTRUCT DERIVED VARIABLES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Infrastructure-adjusted TFP with alternative private R&D elasticity
DATAQ.ln_tfp_util_adj2_prd_adj1 = DATAQ.ln_tfp_util_prd_adj1 - 0.08*DATAQ.ln_rK_GNnonRD;
DATAQ.ln_tfp_util_adj2_prd_adj2 = DATAQ.ln_tfp_util_prd_adj2 - 0.08*DATAQ.ln_rK_GNnonRD;
DATAQ.ln_tfp_util_adj2_prd_adj3 = DATAQ.ln_tfp_util_prd_adj3 - 0.08*DATAQ.ln_rK_GNnonRD;
DATAQ.ln_tfp_util_adj2_prd_adj4 = DATAQ.ln_tfp_util_prd_adj4 - 0.08*DATAQ.ln_rK_GNnonRD;

DATAQ.dtfp_util_adj2_prd_adj1 = DATAQ.ln_tfp_util_adj2_prd_adj1 - lagmatrix(DATAQ.ln_tfp_util_adj2_prd_adj1, 1);
DATAQ.dtfp_util_adj2_prd_adj2 = DATAQ.ln_tfp_util_adj2_prd_adj2 - lagmatrix(DATAQ.ln_tfp_util_adj2_prd_adj2, 1);
DATAQ.dtfp_util_adj2_prd_adj3 = DATAQ.ln_tfp_util_adj2_prd_adj3 - lagmatrix(DATAQ.ln_tfp_util_adj2_prd_adj3, 1);
DATAQ.dtfp_util_adj2_prd_adj4 = DATAQ.ln_tfp_util_adj2_prd_adj4 - lagmatrix(DATAQ.ln_tfp_util_adj2_prd_adj4, 1);

% Investment intensity for rate of return
DATAQ.rint_GRD = (DATAQ.rK_GRD - lagmatrix(DATAQ.rK_GRD, 1)) ./ DATAQ.rGDP;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 3. LOOP OVER PRIVATE R&D ELASTICITY ASSUMPTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

varsy_ela = {'ln_tfp_util_adj2', 'ln_tfp_util_adj2_prd_adj1', ...
             'ln_tfp_util_adj2_prd_adj2', 'ln_tfp_util_adj2_prd_adj3', ...
             'ln_tfp_util_adj2_prd_adj4'};

varsy_ror = {'dtfp_util_adj2', 'dtfp_util_adj2_prd_adj1', ...
             'dtfp_util_adj2_prd_adj2', 'dtfp_util_adj2_prd_adj3', ...
             'dtfp_util_adj2_prd_adj4'};

% Preallocate
ELA_est = NaN(N_priv,1);  ELA_klm_lo = NaN(N_priv,1);  ELA_klm_hi = NaN(N_priv,1);
GRR_est = NaN(N_priv,1);  GRR_klm_lo = NaN(N_priv,1);  GRR_klm_hi = NaN(N_priv,1);

for iter = 1:N_priv

    % --- Panel (a): Elasticity ---
    RES = run_SPIV_ela(varsy_ela{iter}, {'ln_rK_GRD'}, {'z_NDEF'}, {}, ...
        Lcontrols_ND, {}, [-1 2], horizons, p_lags, clevel, DATAQ);
    ELA_est(iter)    = RES.estimate;
    ELA_klm_lo(iter) = RES.klm_lower(2);
    ELA_klm_hi(iter) = RES.klm_upper(2);

    % --- Panel (b): Gross Rate of Return ---
    RES = run_SPIV_ror(varsy_ror{iter}, {'rint_GRD'}, {'z_NDEF'}, {}, ...
        Lcontrols_ND, {}, [-2 5], horizons, p_lags, clevel, DATAQ);
    GRR_est(iter)    = RES.estimate;
    GRR_klm_lo(iter) = RES.klm_lower(2);
    GRR_klm_hi(iter) = RES.klm_upper(2);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 4. FIGURE E5a — ELASTICITY
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure
hold on
b = bar(grid, ELA_est);
er = errorbar(grid, ELA_est, abs(ELA_klm_lo - ELA_est), abs(ELA_klm_hi - ELA_est));
er.Color = [0 0 0];
er.LineStyle = 'none';
xlim([-0.25 2.25])
ylim([0 0.35])
hold off
xaxisproperties = get(gca, 'XAxis');
xaxisproperties.TickLabelInterpreter = 'latex';
yaxisproperties = get(gca, 'YAxis');
yaxisproperties.TickLabelInterpreter = 'latex';
ylabel('PF Elasticity $\hat \phi$', 'Interpreter', 'Latex')
xticklabels(cellstr(strcat(num2str((grid+1)'), '$\times\xi$'))')
xtips1 = b(1).XEndPoints - 0.05;
ytips1 = b(1).YEndPoints;
text(xtips1, ytips1, string(round(b(1).YData, 2)), 'HorizontalAlignment', 'right', ...
    'VerticalAlignment', 'bottom', 'Interpreter', 'Latex')
tightfig;
saveas(gcf, 'figures/Figure_E5a', 'png');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 5. FIGURE E5b — GROSS RATE OF RETURN
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure
hold on
b = bar(grid, GRR_est);
er = errorbar(grid, GRR_est, abs(GRR_klm_lo - GRR_est), abs(GRR_klm_hi - GRR_est));
er.Color = [0 0 0];
er.LineStyle = 'none';
xlim([-0.25 2.25])
ylim([0 4])
hold off
xaxisproperties = get(gca, 'XAxis');
xaxisproperties.TickLabelInterpreter = 'latex';
yaxisproperties = get(gca, 'YAxis');
yaxisproperties.TickLabelInterpreter = 'latex';
ylabel('Gross Rate of Return $\hat \rho$', 'Interpreter', 'Latex')
xticklabels(cellstr(strcat(num2str((grid+1)'), '$\times\xi$'))')
xtips1 = b(1).XEndPoints - 0.05;
ytips1 = b(1).YEndPoints;
text(xtips1, ytips1, string(round(b(1).YData, 2)), 'HorizontalAlignment', 'right', ...
    'VerticalAlignment', 'bottom', 'Interpreter', 'Latex')
tightfig;
saveas(gcf, 'figures/Figure_E5b', 'png');

disp('Figure E.5 generated successfully.');
warning on

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% LOCAL FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function RES = run_SPIV_ela(varsy, varsY, varsZ, varsZ_scale, ...
                             Lcontrols, Ccontrols, range, sel_hors, p, clevel, DATA)
% SP-IV estimation for elasticity with single endogenous variable (K=1).
% Log-level specification: 4-quarter moving average of both y and Y levels.

    K  = length(varsY);
    H  = length(sel_hors);
    Nz = length(varsZ);
    sel_hors_all = 0:max(sel_hors);

    if ~isempty(varsZ_scale)
        ins_names = strcat(varsZ, '_', varsZ_scale);
        ins_data = DATA(:,varsZ).Variables ./ lagmatrix(DATA(:,varsZ_scale).Variables, 1);
        D = array2timetable(ins_data, 'RowTimes', DATA.Time);
        D.Properties.VariableNames = ins_names;
        DATA = synchronize(D, DATA);
    else
        ins_names = varsZ;
    end

    all_Lcontrols = [{varsy} Lcontrols];

    y  = DATA(:, {varsy}).Variables';
    Y  = DATA(:, varsY).Variables';
    XL = lagmatrix(DATA(:, all_Lcontrols).Variables, 1:p)';
    XC = DATA(:, Ccontrols).Variables';
    Z  = DATA(:, ins_names).Variables';

    yHt = (lagmatrix(y',-sel_hors_all)' + lagmatrix(y',-sel_hors_all+1)' + ...
           lagmatrix(y',-sel_hors_all+2)' + lagmatrix(y',-sel_hors_all+3)') / 4;
    yH  = yHt(1+sel_hors, :);
    YH = [];
    for k = 1:K
        YHk = (lagmatrix(Y(k,:)',-sel_hors_all)' + lagmatrix(Y(k,:)',-sel_hors_all+1)' + ...
               lagmatrix(Y(k,:)',-sel_hors_all+2)' + lagmatrix(Y(k,:)',-sel_hors_all+3)') / 4;
        YH = [YH; YHk(1+sel_hors, :)];
    end

    sel = ~isnan(sum([yH; YH; XC; XL; Z]));
    yH=yH(:,sel); YH=YH(:,sel); XL=XL(:,sel); XC=XC(:,sel); Z=Z(:,sel);
    T = size(yH, 2);

    X  = [XC; XL; ones(1,T)];
    Nx = size(X, 1);
    yHo = yH - (yH/X)*X;
    YHo = YH - (YH/X)*X;
    Zo  = Z  - (Z/X)*X;

    Sig_Zo = Zo*Zo'/T;
    irfsy  = (yHo/Zo) * Sig_Zo^0.5;
    irfsY  = (YHo/Zo) * Sig_Zo^0.5;

    IH = eye(H);
    bThetay = reshape(irfsy, 1, H*Nz)';
    bThetaY = [];
    for k = 1:K
        bThetaY(:,k) = reshape(irfsY(1+(k-1)*H:k*H, :), H*Nz, 1);
    end
    b = bThetay' * bThetaY / (bThetaY' * bThetaY);

    Ngrid = 201;
    bgrid = linspace(range(1), range(2), Ngrid);
    S_KLM   = zeros(1, Ngrid);
    OBJ_KLM = zeros(1, Ngrid);
    for c = 1:Ngrid
        uHo_c = yHo - kron(bgrid(c), eye(H)) * YHo;
        S_KLM(c)   = GetKstat(uHo_c, Zo, YHo, Nx, H, K);
        OBJ_KLM(c) = GetARObj(uHo_c, Zo, YHo);
    end

    bmin = bgrid(OBJ_KLM == min(OBJ_KLM));
    fun1 = @(x) GetARObj(yHo - kron(x,eye(H))*YHo, Zo, YHo);
    options = optimoptions('fminunc', 'Display', 'off');
    [bmin,~,~] = fminunc(fun1, bmin, options);

    cil = NaN(1, length(clevel));
    ciu = NaN(1, length(clevel));
    for jj = 1:length(clevel)
        g = S_KLM - chi2inv(clevel(jj), K);
        goi = unique([1; find(sign(g')~=sign(lagmatrix(g',1))); length(bgrid)]);
        gp_est = find(abs(bmin-bgrid) == min(abs(bmin-bgrid)));

        goi_l = goi(goi < gp_est);
        if ~isempty(goi_l)
            bg_l = goi_l(knnsearch(goi_l, gp_est));
        else
            bg_l = 1;
        end
        goi_u = goi(goi > gp_est);
        if ~isempty(goi_u)
            bg_u = goi_u(knnsearch(goi_u, gp_est));
        else
            bg_u = length(bgrid);
        end

        bg = [bg_l bg_u];
        bg_bounds = [min(bgrid(max(bg,1))) max(bgrid(min(bg,length(g))))];
        hit_boundary = [bgrid(1)==bg_bounds(1) bgrid(end)==bg_bounds(2)];
        local_min_check = sum(S_KLM(bgrid>bg_bounds(1) & bgrid<bg_bounds(2)) > chi2inv(clevel(jj),K)) == 0;

        fun = @(x) GetKstat(yHo-kron(x,eye(H))*YHo, Zo, YHo, Nx, H, K) - chi2inv(clevel(jj),K);

        if hit_boundary(1)==0 && local_min_check==1
            cil(jj) = fzero(fun, [bgrid(min(bg)-1) bgrid(min(bg)+1)]);
        elseif hit_boundary(1)==1
            cil(jj) = bg_bounds(1);
        end

        if hit_boundary(2)==0 && local_min_check==1
            ciu(jj) = fzero(fun, [bgrid(max(bg)-1) bgrid(max(bg)+1)]);
        elseif hit_boundary(2)==1
            ciu(jj) = bg_bounds(2);
        end
    end

    RES.estimate   = b;
    RES.klm_lower  = cil;
    RES.klm_upper  = ciu;
end


function RES = run_SPIV_ror(varsy, varsY, varsZ, varsZ_scale, ...
                             Lcontrols, Ccontrols, range, sel_hors, p, clevel, DATA)
% SP-IV estimation for rate of return with single endogenous variable (K=1).
% Both y (cumulated TFP change) and Y (cumulated K/Y change) use
% nancumsum + 4-quarter moving average.

    K  = length(varsY);
    H  = length(sel_hors);
    Nz = length(varsZ);
    sel_hors_all = 0:max(sel_hors);

    if ~isempty(varsZ_scale)
        ins_names = strcat(varsZ, '_', varsZ_scale);
        ins_data = DATA(:,varsZ).Variables ./ lagmatrix(DATA(:,varsZ_scale).Variables, 1);
        D = array2timetable(ins_data, 'RowTimes', DATA.Time);
        D.Properties.VariableNames = ins_names;
        DATA = synchronize(D, DATA);
    else
        ins_names = varsZ;
    end

    all_Lcontrols = [{varsy} Lcontrols];

    y  = DATA(:, {varsy}).Variables';
    Y  = DATA(:, varsY).Variables';
    XL = lagmatrix(DATA(:, all_Lcontrols).Variables, 1:p)';
    XC = DATA(:, Ccontrols).Variables';
    Z  = DATA(:, ins_names).Variables';

    yHt = (nancumsum(lagmatrix(y',-sel_hors_all),2)' + ...
           nancumsum(lagmatrix(y',-sel_hors_all+1),2)' + ...
           nancumsum(lagmatrix(y',-sel_hors_all+2),2)' + ...
           nancumsum(lagmatrix(y',-sel_hors_all+3),2)') / 4;
    yH  = yHt(1+sel_hors, :);

    YH = [];
    for k = 1:K
        YHk = (nancumsum(lagmatrix(Y(k,:)',-sel_hors_all),2)' + ...
               nancumsum(lagmatrix(Y(k,:)',-sel_hors_all+1),2)' + ...
               nancumsum(lagmatrix(Y(k,:)',-sel_hors_all+2),2)' + ...
               nancumsum(lagmatrix(Y(k,:)',-sel_hors_all+3),2)') / 4;
        YH = [YH; YHk(1+sel_hors, :)];
    end

    sel = ~isnan(sum([yH; YH; XC; XL; Z]));
    yH=yH(:,sel); YH=YH(:,sel); XL=XL(:,sel); XC=XC(:,sel); Z=Z(:,sel);
    T = size(yH, 2);

    X  = [XC; XL; ones(1,T)];
    Nx = size(X, 1);
    yHo = yH - (yH/X)*X;
    YHo = YH - (YH/X)*X;
    Zo  = Z  - (Z/X)*X;

    Sig_Zo = Zo*Zo'/T;
    irfsy  = (yHo/Zo) * Sig_Zo^0.5;
    irfsY  = (YHo/Zo) * Sig_Zo^0.5;

    IH = eye(H);
    bThetay = reshape(irfsy, 1, H*Nz)';
    bThetaY = [];
    for k = 1:K
        bThetaY(:,k) = reshape(irfsY(1+(k-1)*H:k*H, :), H*Nz, 1);
    end
    b = bThetay' * bThetaY / (bThetaY' * bThetaY);

    Ngrid = 201;
    bgrid = linspace(range(1), range(2), Ngrid);
    S_KLM   = zeros(1, Ngrid);
    OBJ_KLM = zeros(1, Ngrid);
    for c = 1:Ngrid
        uHo_c = yHo - kron(bgrid(c), eye(H)) * YHo;
        S_KLM(c)   = GetKstat(uHo_c, Zo, YHo, Nx, H, K);
        OBJ_KLM(c) = GetARObj(uHo_c, Zo, YHo);
    end

    bmin = bgrid(OBJ_KLM == min(OBJ_KLM));
    fun1 = @(x) GetARObj(yHo - kron(x,eye(H))*YHo, Zo, YHo);
    options = optimoptions('fminunc', 'Display', 'off');
    [bmin,~,~] = fminunc(fun1, bmin, options);

    cil = NaN(1, length(clevel));
    ciu = NaN(1, length(clevel));
    for jj = 1:length(clevel)
        g = S_KLM - chi2inv(clevel(jj), K);
        goi = unique([1; find(sign(g')~=sign(lagmatrix(g',1))); length(bgrid)]);
        gp_est = find(abs(bmin-bgrid) == min(abs(bmin-bgrid)));

        goi_l = goi(goi < gp_est);
        if ~isempty(goi_l)
            bg_l = goi_l(knnsearch(goi_l, gp_est));
        else
            bg_l = 1;
        end
        goi_u = goi(goi > gp_est);
        if ~isempty(goi_u)
            bg_u = goi_u(knnsearch(goi_u, gp_est));
        else
            bg_u = length(bgrid);
        end

        bg = [bg_l bg_u];
        bg_bounds = [min(bgrid(max(bg,1))) max(bgrid(min(bg,length(g))))];
        hit_boundary = [bgrid(1)==bg_bounds(1) bgrid(end)==bg_bounds(2)];
        local_min_check = sum(S_KLM(bgrid>bg_bounds(1) & bgrid<bg_bounds(2)) > chi2inv(clevel(jj),K)) == 0;

        fun = @(x) GetKstat(yHo-kron(x,eye(H))*YHo, Zo, YHo, Nx, H, K) - chi2inv(clevel(jj),K);

        if hit_boundary(1)==0 && local_min_check==1
            cil(jj) = fzero(fun, [bgrid(min(bg)-1) bgrid(min(bg)+1)]);
        elseif hit_boundary(1)==1
            cil(jj) = bg_bounds(1);
        end

        if hit_boundary(2)==0 && local_min_check==1
            ciu(jj) = fzero(fun, [bgrid(max(bg)-1) bgrid(max(bg)+1)]);
        elseif hit_boundary(2)==1
            ciu(jj) = bg_bounds(2);
        end
    end

    RES.estimate   = b;
    RES.klm_lower  = cil;
    RES.klm_upper  = ciu;
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
