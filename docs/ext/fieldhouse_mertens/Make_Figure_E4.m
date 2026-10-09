%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Make_Figure_E4.m
%
% Generates Appendix Figure E.4: Nondefense Government R&D, Elasticity
% and Return Estimates Assuming Different Depreciation Rates
%
% Panel (a): SP-IV production function elasticity (phi_hat) at five
%            depreciation rate scalings: 0, 0.5, 1, 1.5, 2 times BEA delta.
% Panel (b): SP-IV net rate of return (rho_hat - x*delta) at the same
%            depreciation rate scalings.
%
% For each depreciation scaling, the government R&D capital stock is
% reconstructed from BEA source data using the scaled depreciation rate,
% then SP-IV is re-estimated.
%
% Output:  figures/Figure_E4a.png  (elasticity bar chart)
%          figures/Figure_E4b.png  (net rate of return bar chart)
%
% Source:  DATAQ timetable + data_folder (must be in workspace)
% Requires: btrace.m, nancumsum.m, chainsum.m, tightfig.m (auxiliary_files/)
%           knnsearch (Statistics and Machine Learning Toolbox)
%           fminunc   (Optimization Toolbox)
%           smoothdata (MATLAB R2017a+)
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

dep_grid = [0 0.5 1 1.5 2];
N_dep    = length(dep_grid);

% Preallocate
ELA_est = NaN(N_dep,1);  ELA_klm_lo = NaN(N_dep,1);  ELA_klm_hi = NaN(N_dep,1);
ROR_est = NaN(N_dep,1);  ROR_klm_lo = NaN(N_dep,1);  ROR_klm_hi = NaN(N_dep,1);
avg_dep = NaN(N_dep,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2. LOOP OVER DEPRECIATION SCALINGS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for iter = 1:N_dep
    dep_scale = dep_grid(iter);
   % fprintf('Depreciation scale = %.1f x delta\n', dep_scale);

    % --- Reconstruct capital stock with alternative depreciation ---
    [dataq, avg_dep(iter)] = Get_GRD_capital_stock_alt_depreciation(dep_scale, data_folder);

    % Rescale to match baseline initial value, update DATAQ copy
    DATAQ2 = DATAQ;
    dataq.rK_GRD = dataq.rK_GRD ./ dataq.rK_GRD(1) * DATAQ.rK_GRD(1);
    DATAQ2(:,{'rK_GRD'}) = dataq(:,{'rK_GRD'});
    DATAQ2.ln_rK_GRD = log(DATAQ2.rK_GRD);
    DATAQ2.rint_GRD   = (DATAQ2.rK_GRD - lagmatrix(DATAQ2.rK_GRD,1)) ./ DATAQ2.rGDP;

    % --- Panel (a): Elasticity ---
    RES = run_SPIV_ela('ln_tfp_util_adj2', {'ln_rK_GRD'}, {'z_NDEF'}, {}, ...
        Lcontrols_ND, {}, [-1 2], horizons, p_lags, clevel, DATAQ2);
    ELA_est(iter)    = RES.estimate;
    ELA_klm_lo(iter) = RES.klm_lower(2);
    ELA_klm_hi(iter) = RES.klm_upper(2);

    % --- Panel (b): Rate of Return ---
    RES = run_SPIV_ror('dtfp_util_adj2', {'rint_GRD'}, {'z_NDEF'}, {}, ...
        Lcontrols_ND, {}, [-2 5], horizons, p_lags, clevel, DATAQ2);
    ROR_est(iter)    = RES.estimate;
    ROR_klm_lo(iter) = RES.klm_lower(2);
    ROR_klm_hi(iter) = RES.klm_upper(2);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 3. FIGURE E4a — ELASTICITY
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure
hold on
b = bar(dep_grid, ELA_est);
er = errorbar(dep_grid, ELA_est, abs(ELA_klm_lo - ELA_est), abs(ELA_klm_hi - ELA_est));
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
xticklabels(cellstr(strcat(num2str(dep_grid'), '$\times\delta$'))')
xtips1 = b(1).XEndPoints - 0.05;
ytips1 = b(1).YEndPoints;
text(xtips1, ytips1, string(round(b(1).YData, 2)), 'HorizontalAlignment', 'right', ...
    'VerticalAlignment', 'bottom', 'Interpreter', 'Latex')
tightfig;
saveas(gcf, 'figures/Figure_E4a', 'png');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 4. FIGURE E4b — NET RATE OF RETURN
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure
hold on
b = bar(dep_grid, ROR_est - avg_dep);
er = errorbar(dep_grid, ROR_est, abs(ROR_klm_lo - ROR_est), abs(ROR_klm_hi - ROR_est));
er.Color = [0 0 0];
er.LineStyle = 'none';
xlim([-0.25 2.25])
ylim([0 4])
hold off
xaxisproperties = get(gca, 'XAxis');
xaxisproperties.TickLabelInterpreter = 'latex';
yaxisproperties = get(gca, 'YAxis');
yaxisproperties.TickLabelInterpreter = 'latex';
ylabel('Net Rate of Return $\hat \rho-(x \times \delta) $', 'Interpreter', 'Latex')
xticklabels(cellstr(strcat(num2str(dep_grid'), '$\times\delta$'))')
xtips1 = b(1).XEndPoints - 0.05;
ytips1 = b(1).YEndPoints;
text(xtips1, ytips1, string(round(b(1).YData, 2)), 'HorizontalAlignment', 'right', ...
    'VerticalAlignment', 'bottom', 'Interpreter', 'Latex')
tightfig;
saveas(gcf, 'figures/Figure_E4b', 'png');

disp('Figure E.4 generated successfully.');
warning on

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% LOCAL FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function RES = run_SPIV_ela(varsy, varsY, varsZ, varsZ_scale, ...
                             Lcontrols, Ccontrols, range, sel_hors, p, clevel, DATA)
% SP-IV estimation for elasticity with single endogenous variable (K=1).
% Log-level specification: 4-quarter moving average of both y and Y levels.
% Returns point estimate and KLM confidence intervals.

    K  = length(varsY);
    H  = length(sel_hors);
    Nz = length(varsZ);
    sel_hors_all = 0:max(sel_hors);

    % Construct instruments
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

    % Multi-horizon: 4Q moving average of levels
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

    PYHo   = (YHo/Zo)*Zo;
    Sig_Zo = Zo*Zo'/T;
    irfsy  = (yHo/Zo) * Sig_Zo^0.5;
    irfsY  = (YHo/Zo) * Sig_Zo^0.5;

    IH = eye(H);
    R  = kron(eye(K), IH(:));
    bThetay = reshape(irfsy, 1, H*Nz)';
    bThetaY = [];
    for k = 1:K
        bThetaY(:,k) = reshape(irfsY(1+(k-1)*H:k*H, :), H*Nz, 1);
    end
    b = bThetay' * bThetaY / (bThetaY' * bThetaY);

    % KLM confidence intervals (1D grid inversion)
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
% nancumsum + 4-quarter moving average, since both are first-differences.

    K  = length(varsY);
    H  = length(sel_hors);
    Nz = length(varsZ);
    sel_hors_all = 0:max(sel_hors);

    % Construct instruments
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

    % Multi-horizon: nancumsum + 4Q MA for both y and Y
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

    PYHo   = (YHo/Zo)*Zo;
    Sig_Zo = Zo*Zo'/T;
    irfsy  = (yHo/Zo) * Sig_Zo^0.5;
    irfsY  = (YHo/Zo) * Sig_Zo^0.5;

    IH = eye(H);
    R  = kron(eye(K), IH(:));
    bThetay = reshape(irfsy, 1, H*Nz)';
    bThetaY = [];
    for k = 1:K
        bThetaY(:,k) = reshape(irfsY(1+(k-1)*H:k*H, :), H*Nz, 1);
    end
    b = bThetay' * bThetaY / (bThetaY' * bThetaY);

    % KLM confidence intervals (1D grid inversion)
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


function [DATAQ, avg_dep] = Get_GRD_capital_stock_alt_depreciation(depr_scale, data_folder)
% Construct government R&D capital stock with scaled depreciation rates.
%
% depr_scale: multiplication factor for depreciation (0, 0.5, 1, 1.5, 2)
% Returns DATAQ timetable with rK_GRD and the average depreciation rate.

    lastdatapointA = datetime(2021,12,1);
    lastdatapointQ = datetime(2022,8,1);
    ATime = (datetime(1925,1,1):calmonths(12):lastdatapointA)';
    QTime = (datetime(1947,1,1):calmonths(3):lastdatapointQ)';

    % Annual BEA Fixed Asset Accounts (FAA) data
    FAA_7_1 = readtable(strcat(data_folder,'BEA/FAA_7_1.csv'));
    nipa_table_lines = FAA_7_1(:,1).Variables;
    nK = FAA_7_1(~isnan(nipa_table_lines),3:end).Variables';

    FAA_7_2 = readtable(strcat(data_folder,'BEA/FAA_7_2.csv'));
    nipa_table_lines = FAA_7_2(:,1).Variables;
    rKqi = FAA_7_2(~isnan(nipa_table_lines),3:end).Variables';
    rK = rKqi/100 .* nK(year(ATime)==2012,:);
    rKL = lagmatrix(rK,1);

    FAA_7_3 = readtable(strcat(data_folder,'BEA/FAA_7_3.csv'));
    nipa_table_lines = FAA_7_3(:,1).Variables;
    nDEP = FAA_7_3(~isnan(nipa_table_lines),3:end).Variables';

    FAA_7_4 = readtable(strcat(data_folder,'BEA/FAA_7_4.csv'));
    nipa_table_lines = FAA_7_4(:,1).Variables;
    rDEPqi = FAA_7_4(~isnan(nipa_table_lines),3:end).Variables';
    rDEP = rDEPqi/100 .* nDEP(year(ATime)==2012,:);

    FAA_7_5 = readtable(strcat(data_folder,'BEA/FAA_7_5.csv'));
    nipa_table_lines = FAA_7_5(:,1).Variables;
    nI = FAA_7_5(~isnan(nipa_table_lines),27:end).Variables';

    FAA_7_6 = readtable(strcat(data_folder,'BEA/FAA_7_6.csv'));
    nipa_table_lines = FAA_7_6(:,1).Variables;
    rIqi = FAA_7_6(~isnan(nipa_table_lines),27:end).Variables';
    rI = rIqi/100 .* nI(year(ATime)==2012,:);

    vars  = {'GEQI','GSTR','GRD','FDEQI','FDSTR','FDRD','FNEQI','FNSTR','FNRD','SLEQI','SLSTR','SLRD'};
    lines = [2 3 18 23 30 35 39 40 52 56 57 72];
    sely  = ATime > datetime(1946,12,1);

    D = array2timetable([nK(sely,lines) rK(sely,lines) rKL(sely,lines) rDEP(sely,lines) rI(sely,lines) nI(sely,lines)]*1000, 'RowTimes', dateshift(ATime(sely),'end','year'));
    D.Properties.VariableNames = [strcat('nKA_',vars) strcat('rKA_',vars) strcat('rKAL_',vars) strcat('rDEP_',vars) strcat('rIA_',vars) strcat('nIA_',vars)];
    KDATA = retime(D, 'quarterly', 'mean');

    % Quarterly NIPA investment data (NIPA 3.9.3, 3.9.5)
    NIPA_3_9_5 = readtable(strcat(data_folder,'BEA/NIPA_3_9_5'));
    nipa_table_lines = NIPA_3_9_5(:,1).Variables;
    nGOVCI = NIPA_3_9_5(~isnan(nipa_table_lines),3:end).Variables';

    NIPA_3_9_3 = readtable(strcat(data_folder,'BEA/NIPA_3_9_3'));
    nipa_table_lines = NIPA_3_9_3(:,1).Variables;
    rGOVCI = NIPA_3_9_3(~isnan(nipa_table_lines),3:end).Variables';
    rGOVCI = rGOVCI./100 .* mean(nGOVCI(year(QTime)==2012,:));

    qlines = [5 4 6 21 20 22 29 28 30 37 36 38];
    IDATA = array2timetable([rGOVCI(:,qlines) nGOVCI(:,qlines)], 'RowTimes', QTime);
    IDATA.Properties.VariableNames = [strcat('rI_',vars) strcat('nI_',vars)];

    DATA = synchronize(KDATA, IDATA);

    qlines2 = 8;
    IDATA2 = array2timetable([rGOVCI(:,qlines2) nGOVCI(:,qlines2)], 'RowTimes', QTime);
    IDATA2.Properties.VariableNames = {'rI_Grd', 'nI_Grd'};
    DATA = synchronize(DATA, IDATA2);

    DATA.rFDGOV = rGOVCI(:,17);

    % Construct quarterly capital stocks for R&D components
    rd_vars = {'FDRD', 'FNRD', 'SLRD'};

    K0 = DATA(DATA.Time==datetime(1947,10,1), strcat('rKAL_',rd_vars)).Variables;
    x  = DATA(:, strcat('rI_',rd_vars)).Variables / 4;

    % Compute depreciation rates from annual FAA data
    dd = zeros(size(x));
    for t = 1:length(x)
        sel_t = DATA.Time == datetime(year(DATA(t,:).Time), 10, 1);
        if sum(sel_t) ~= 0
            dd(t,:) = DATA(sel_t, strcat('rDEP_',rd_vars)).Variables ./ DATA(sel_t, strcat('rKAL_',rd_vars)).Variables;
        else
            sel_t = DATA.Time == datetime(year(DATA(t,:).Time)-1, 10, 1);
            dd(t,:) = DATA(sel_t, strcat('rDEP_',rd_vars)).Variables ./ DATA(sel_t, strcat('rKAL_',rd_vars)).Variables;
        end
    end

    dds = zeros(size(dd));
    for j = 1:size(dd,2)
        dds(:,j) = min(max(smoothdata(dd(:,j)), 0), 1);
    end

    % Capital accumulation with scaled depreciation
    Kq = zeros(length(x)+1, size(x,2));
    Kq(1,:) = K0;
    dep_store = zeros(length(x), size(x,2));
    for t = 1:length(x)
        dep = (1 - (1 - dds(t,:)*depr_scale).^0.25);
        Kq(t+1,:) = (1 - dep) .* Kq(t,:) + x(t,:);
        dep_store(t,:) = 1 - (1 - dep).^4;
    end
    Kq = Kq(2:end,:);

    % Convert to 2012 dollars
    sel_2012 = DATA.Time == datetime(2012,10,1);
    rKq = Kq ./ mean(Kq(year(DATA.Time)==2012,:)) .* DATA(sel_2012, strcat('nKA_',rd_vars)).Variables;
    pI = DATA(:, strcat('nI_',rd_vars)).Variables ./ DATA(:, strcat('rI_',rd_vars)).Variables;
    nKq = rKq .* (pI ./ mean(pI(year(DATA.Time)==2012,:)));

    Dq = array2timetable([rKq nKq], 'RowTimes', DATA.Time);
    Dq.Properties.VariableNames = [strcat('rK_',rd_vars) strcat('nK_',rd_vars)];
    DATA = synchronize(DATA, Dq);

    % Average depreciation rate (nominal-share-weighted)
    avg_dep = mean(sum(dep_store .* DATA(:, strcat('nK_',rd_vars)).Variables ./ sum(DATA(:, strcat('nK_',rd_vars)).Variables, 2), 2));

    % Chain-weight aggregate GRD capital
    nK_GRD = sum(DATA(:, {'nK_FNRD','nK_FDRD','nK_SLRD'}).Variables, 2);
    pf = DATA(:, {'nI_FNRD','nI_FDRD','nI_SLRD'}).Variables;
    p  = pf ./ (DATA(:, {'rI_FNRD','rI_FDRD','rI_SLRD'}).Variables);
    p(isnan(p)) = 0;
    q = DATA(:, {'rK_FNRD','rK_FDRD','rK_SLRD'}).Variables;
    rK_GRD = chainsum(q, p.*q, year(DATA.Time)==2012);

    Dagg = array2timetable([nK_GRD rK_GRD], 'RowTimes', DATA.Time);
    Dagg.Properties.VariableNames = {'nK_GRD', 'rK_GRD'};
    DATA = synchronize(DATA, Dagg);

    DATAQ = DATA(:, {'rK_GRD'});
    DATAQ = DATAQ(year(DATAQ.Time) < 2022, :);
    DATAQ = DATAQ(year(DATAQ.Time) > 1946, :);
end


function Y = chainsum(rX, nX, base)
% Fisher ideal chain-weighted aggregation
    q  = rX;
    qL = lagmatrix(q, 1);
    p  = abs(nX ./ q);
    p(isnan(p)) = 0;
    pL = lagmatrix(p, 1);

    DY = (sum(q.*p,2) ./ sum(qL.*p,2) .* sum(q.*pL,2) ./ sum(qL.*pL,2)).^0.5;
    DY(1) = 1;
    Y = cumprod(DY);
    Y = Y ./ mean(Y(base)) * mean(sum(nX(base,:),2));
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
