%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Make_Table_2.m
%
% Generates Table 2: Estimates of the Return to Government R&D Capital
%
% This script estimates the rate of return to government R&D capital using
% System Projections on Instrument Variables (SP-IV). The estimator
% projects cumulated TFP changes and cumulated K/Y ratio changes onto
% narrative appropriations shock instruments at multiple horizons, then
% regresses the TFP projections on the capital projections.
%
% (SP-IV = System Projections on Instrumental Variables)
%
% The table reports:
%   BOE  — back-of-envelope rate of return (log-level elasticity / (K/Y))
%   ROR  — SP-IV rate of return estimate with 95% KLM confidence intervals
%
% for three infrastructure TFP adjustments (low, moderate, high).
%
% Rows 1-3: Total government R&D capital (K=1, rint_GRD)
% Rows 4-7: ND/D decomposition (K=2, dKY_GNRD + dKY_FDRD), GNRD reported
%
% Output:  tables/results_Table_2.tex  (LaTeX table body, 7 rows)
%
% Source:  DATAQ timetable (must be in workspace)
% Requires: btrace.m, nancumsum.m (in auxiliary_files/)
%           knnsearch (Statistics and Machine Learning Toolbox)
%           fminunc   (Optimization Toolbox)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
warning off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1. DATA PREPARATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% --- Rate of return dependent variable (Total, rows 1-3) ---
% rint_GRD = change in real GRD capital / real GDP
QQDATA = DATAQ;
QQDATA.rint_GRD = (QQDATA.rK_GRD - lagmatrix(QQDATA.rK_GRD, 1)) ./ QQDATA.rGDP;

% --- Capital-output ratio changes for ND/D decomposition (rows 4-7) ---
% dKY = (rK - lag(rK)) / rGDP for each component
rk_vars = {'rK_GRD', 'rK_GNRD', 'rK_FDRD'};
v = QQDATA(:, rk_vars).Variables;
v = v - lagmatrix(v, 1);
D = array2timetable(v ./ QQDATA.rGDP, 'RowTimes', QQDATA.Time);
D.Properties.VariableNames = {'dKY_GRD', 'dKY_GNRD', 'dKY_FDRD'};
QQDATA = synchronize(QQDATA, D);

% --- Share-weighted capital growth rates for BOE elasticities (rows 4-7) ---
decomp_vars = {'GRD', 'GNRD', 'FDRD'};
nK = QQDATA(:, strcat('nK_', decomp_vars)).Variables;
shares = nK ./ nK(:,1);
shares = 0.5 * (shares + lagmatrix(shares, 1));
dlnrK  = QQDATA(:, strcat('ln_rK_', decomp_vars)).Variables;
dlnrK  = dlnrK - lagmatrix(dlnrK, 1);
D = array2timetable([shares .* dlnrK, shares], 'RowTimes', QQDATA.Time);
D.Properties.VariableNames = [strcat('dln_snK_', decomp_vars), ...
                               strcat('shares_', decomp_vars)];
QQDATA = synchronize(QQDATA, D);

% --- Back-of-envelope scaling ratio ---
% K/Y ratio averaged over 1948Q1-2007Q4
sel = (year(QQDATA.Time) > 1947) & (year(QQDATA.Time) < 2008);
ratio = nanmean(QQDATA.rK_GRD(sel) ./ QQDATA.rGDP(sel));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2. ESTIMATION SETTINGS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clevel   = [0.90 0.95 0.99];   % confidence levels
horizons = 3:4:59;              % every 4th quarter from Q4
p_lags   = 4;                   % number of lags

% TFP variants: low, moderate, high infrastructure adjustment
tfp_level_vars = {'ln_tfp_util_adj1', 'ln_tfp_util_adj2', 'ln_tfp_util_adj3'};
tfp_diff_vars  = {'dtfp_util_adj1',   'dtfp_util_adj2',   'dtfp_util_adj3'};

% Baseline lagged control variables
Lcontrols_ND = {'ln_cum_rALL_NDEF', 'ln_util', 'ln_tfp_util', ...
                'ln_rK_PRD_fernald', 'R_EW_COMB_TECH', ...
                'ln_rK_GRD', 'Military_news_scaled'};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 3. BOE ELASTICITIES (point estimates only, no CIs needed)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

BOE = NaN(7, 3);

% --- BOE Rows 1-3: Total GRD, log-level elasticity ---
% Row 1: Exo ND
for t = 1:3
    BOE(1,t) = get_elasticity(tfp_level_vars{t}, {'ln_rK_GRD'}, ...
        {'z_NDEF'}, {}, Lcontrols_ND, {}, horizons, p_lags, QQDATA) / ratio;
end
% Row 2: Exo ND, No NASA
for t = 1:3
    BOE(2,t) = get_elasticity(tfp_level_vars{t}, {'ln_rK_GRD'}, ...
        {'z_NDEF_exspace'}, {}, Lcontrols_ND, {'z_SPACE'}, horizons, p_lags, QQDATA) / ratio;
end
% Row 3: All ND
for t = 1:3
    BOE(3,t) = get_elasticity(tfp_level_vars{t}, {'ln_rK_GRD'}, ...
        {'z_allNDEF'}, {}, Lcontrols_ND, {}, horizons, p_lags, QQDATA) / ratio;
end

% --- BOE Rows 4-7: ND/D decomposition, GNRD component elasticity ---
% Row 4: Exo ND
for t = 1:3
    BOE(4,t) = get_elasticity_decomp(tfp_level_vars{t}, ...
        {'dln_snK_GNRD','dln_snK_FDRD'}, {'z_NDEF'}, {}, ...
        Lcontrols_ND, {}, horizons, p_lags, QQDATA) / ratio;
end
% Row 5: Exo ND/D
for t = 1:3
    BOE(5,t) = get_elasticity_decomp(tfp_level_vars{t}, ...
        {'dln_snK_GNRD','dln_snK_FDRD'}, {'rEXO_NDEF','rEXO_NSEC'}, ...
        {'rK_FNRD','rK_FDRD'}, Lcontrols_ND, {}, horizons, p_lags, QQDATA) / ratio;
end
% Row 6: Exo ND, No NASA
for t = 1:3
    BOE(6,t) = get_elasticity_decomp(tfp_level_vars{t}, ...
        {'dln_snK_GNRD','dln_snK_FDRD'}, {'z_NDEF_exspace'}, {}, ...
        Lcontrols_ND, {'z_SPACE'}, horizons, p_lags, QQDATA) / ratio;
end
% Row 7: All ND
for t = 1:3
    BOE(7,t) = get_elasticity_decomp(tfp_level_vars{t}, ...
        {'dln_snK_GNRD','dln_snK_FDRD'}, {'z_allNDEF'}, {}, ...
        Lcontrols_ND, {}, horizons, p_lags, QQDATA) / ratio;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 4. ROR ESTIMATES (with KLM confidence intervals)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Preallocate: 7 rows x 3 TFP variants
ROR_est   = NaN(7,3);  ROR_ci  = NaN(7,3,2);  ROR_stars = NaN(7,3);

% --- Rows 1-3: Total GRD (K=1, 1D grid search) ---
% fprintf('Row 1: Total GRD, Exo ND\n');
for t = 1:3
    RES = run_SPIV_ror(tfp_diff_vars{t}, {'rint_GRD'}, {'z_NDEF'}, {}, ...
        Lcontrols_ND, {}, [-2 5], horizons, p_lags, clevel, QQDATA);
    ROR_est(1,t)=RES.estimate; ROR_ci(1,t,:)=[RES.klm_lower(2) RES.klm_upper(2)]; ROR_stars(1,t)=RES.stars_klm;
end

% fprintf('Row 2: Total GRD, Exo ND No NASA\n');
for t = 1:3
    RES = run_SPIV_ror(tfp_diff_vars{t}, {'rint_GRD'}, {'z_NDEF_exspace'}, {}, ...
        Lcontrols_ND, {'z_SPACE'}, [-2 5], horizons, p_lags, clevel, QQDATA);
    ROR_est(2,t)=RES.estimate; ROR_ci(2,t,:)=[RES.klm_lower(2) RES.klm_upper(2)]; ROR_stars(2,t)=RES.stars_klm;
end

% fprintf('Row 3: Total GRD, All ND\n');
for t = 1:3
    RES = run_SPIV_ror(tfp_diff_vars{t}, {'rint_GRD'}, {'z_allNDEF'}, {}, ...
        Lcontrols_ND, {}, [-2 5], horizons, p_lags, clevel, QQDATA);
    ROR_est(3,t)=RES.estimate; ROR_ci(3,t,:)=[RES.klm_lower(2) RES.klm_upper(2)]; ROR_stars(3,t)=RES.stars_klm;
end

% --- Rows 4-7: ND/D decomposition (K=2, 2D grid search) ---
% fprintf('Row 4: ND/D Decomp, Exo ND\n');
for t = 1:3
    RES = run_SPIV_ror_decomp(tfp_diff_vars{t}, {'dKY_GNRD','dKY_FDRD'}, ...
        {'z_NDEF'}, {}, Lcontrols_ND, {}, [-2 5; -2 5], ...
        horizons, p_lags, clevel, QQDATA);
    ROR_est(4,t)=RES.estimate; ROR_ci(4,t,:)=[RES.klm_lower(2) RES.klm_upper(2)]; ROR_stars(4,t)=RES.stars_klm;
end

% fprintf('Row 5: ND/D Decomp, Exo ND/D\n');
for t = 1:3
    RES = run_SPIV_ror_decomp(tfp_diff_vars{t}, {'dKY_GNRD','dKY_FDRD'}, ...
        {'rEXO_NDEF','rEXO_NSEC'}, {'rK_FNRD','rK_FDRD'}, ...
        Lcontrols_ND, {}, [-2 5; -2 5], horizons, p_lags, clevel, QQDATA);
    ROR_est(5,t)=RES.estimate; ROR_ci(5,t,:)=[RES.klm_lower(2) RES.klm_upper(2)]; ROR_stars(5,t)=RES.stars_klm;
end

% fprintf('Row 6: ND/D Decomp, Exo ND No NASA\n');
for t = 1:3
    RES = run_SPIV_ror_decomp(tfp_diff_vars{t}, {'dKY_GNRD','dKY_FDRD'}, ...
        {'z_NDEF_exspace'}, {}, Lcontrols_ND, {'z_SPACE'}, [-2 5; -2 5], ...
        horizons, p_lags, clevel, QQDATA);
    ROR_est(6,t)=RES.estimate; ROR_ci(6,t,:)=[RES.klm_lower(2) RES.klm_upper(2)]; ROR_stars(6,t)=RES.stars_klm;
end

% fprintf('Row 7: ND/D Decomp, All ND\n');
for t = 1:3
    RES = run_SPIV_ror_decomp(tfp_diff_vars{t}, {'dKY_GNRD','dKY_FDRD'}, ...
        {'z_allNDEF'}, {}, Lcontrols_ND, {}, [-2 5; -2 5], ...
        horizons, p_lags, clevel, QQDATA);
    ROR_est(7,t)=RES.estimate; ROR_ci(7,t,:)=[RES.klm_lower(2) RES.klm_upper(2)]; ROR_stars(7,t)=RES.stars_klm;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 5. FORMAT AND WRITE LaTeX TABLE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

label1 = {'Total','Total','Total','ND/D','ND/D','ND/D','ND/D'};
label2 = {'Exo ND','Exo ND, No NASA','All ND', ...
           'Exo ND','Exo ND/D','Exo ND, No NASA','All ND'};

fileID = fopen('tables/results_Table_2.tex', 'w');
for row = 1:7
    % Row number, decomposition label, instrument label
    fprintf(fileID, '\\footnotesize{$[%6.0f]$}&', row);
    fprintf(fileID, '\\footnotesize{%s}', label1{row});
    fprintf(fileID, '&');
    fprintf(fileID, '\\footnotesize{%s}', label2{row});

    % --- Moderate adjustment (columns 4-5) ---
    fprintf(fileID, '& \\footnotesize{$%6.2f$}', BOE(row,2));
    write_cell(fileID, ROR_est(row,2), squeeze(ROR_ci(row,2,:)), ROR_stars(row,2));

    fprintf(fileID, '&  ');  % blank separator

    % --- Low adjustment (columns 7-8) — stars from moderate ---
    fprintf(fileID, '& \\footnotesize{$%6.2f$}', BOE(row,1));
    write_cell(fileID, ROR_est(row,1), squeeze(ROR_ci(row,1,:)), ROR_stars(row,2));

    fprintf(fileID, '&  ');  % blank separator

    % --- High adjustment (columns 10-11) — own stars ---
    fprintf(fileID, '& \\footnotesize{$%6.2f$}', BOE(row,3));
    write_cell(fileID, ROR_est(row,3), squeeze(ROR_ci(row,3,:)), ROR_stars(row,3));

    fprintf(fileID, '\\\\ \n');
end
fclose(fileID);

disp('Table 2 generated successfully: tables/results_Table_2.tex');
warning on

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% LOCAL FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function write_cell(fid, est, ci, nstars)
% Write one table cell: ROR estimate with 95% CI underneath, significance stars
    fprintf(fid, '&\\footnotesize{ $ \\underset{\\tiny{(%6.2f,%6.2f)}}{%6.2f ', ...
            ci(1), ci(2), est);
    if nstars == 3
        fprintf(fid, '^{\\ast\\ast\\ast} } $ }');
    elseif nstars == 2
        fprintf(fid, '^{\\ast\\ast} } $ }');
    elseif nstars == 1
        fprintf(fid, '^{\\ast} } $ }');
    else
        fprintf(fid, '} $ }');
    end
end


function b = get_elasticity(varsy, varsY, varsZ, varsZ_scale, ...
                             Lcontrols, Ccontrols, sel_hors, p, DATA)
% SP-IV point estimate with single endogenous variable (K=1).
% Log-level specification: 4-quarter moving average of levels.
% Used only for BOE computation (no CIs needed).

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


function b1 = get_elasticity_decomp(varsy, varsY, varsZ, varsZ_scale, ...
                                     Lcontrols, Ccontrols, sel_hors, p, DATA)
% SP-IV point estimate with two endogenous variables (K=2).
% Returns GNRD (component 1) elasticity only.
% Log-level specification for TFP; nancumsum for capital growth rates.

    K  = length(varsY);
    H  = length(sel_hors);
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
    Nz = length(ins_names);

    all_Lcontrols = unique([{varsy} Lcontrols]);

    y  = DATA(:, {varsy}).Variables';
    Y  = DATA(:, varsY).Variables';
    XL = lagmatrix(DATA(:, all_Lcontrols).Variables, 1:p)';
    XC = DATA(:, Ccontrols).Variables';
    Z  = DATA(:, ins_names).Variables';

    % TFP: 4Q moving average of levels
    yHt = (lagmatrix(y',-sel_hors_all)' + lagmatrix(y',-sel_hors_all+1)' + ...
           lagmatrix(y',-sel_hors_all+2)' + lagmatrix(y',-sel_hors_all+3)') / 4;
    yH = yHt(1+sel_hors, :);

    % Capital growth: cumulate then 4Q MA
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
    b1 = b(1);  % GNRD component only
end


function RES = run_SPIV_ror(varsy, varsY, varsZ, varsZ_scale, ...
                             Lcontrols, Ccontrols, range, sel_hors, p, clevel, DATA)
% SP-IV estimation for rate of return with single endogenous variable (K=1).
% Used for rows 1-3 of Table 2.
%
% Both y (cumulated TFP change) and Y (cumulated K/Y change) use
% nancumsum + 4-quarter moving average, since both are first-differences.
% Includes 1D KLM grid search for confidence intervals.

    K  = length(varsY);
    H  = length(sel_hors);
    Nz = length(varsZ);
    sel_hors_all = 0:max(sel_hors);

    % --- Construct instruments ---
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

    % --- Extract data matrices ---
    y  = DATA(:, {varsy}).Variables';
    Y  = DATA(:, varsY).Variables';
    XL = lagmatrix(DATA(:, all_Lcontrols).Variables, 1:p)';
    XC = DATA(:, Ccontrols).Variables';
    Z  = DATA(:, ins_names).Variables';

    % --- Multi-horizon outcomes: nancumsum + 4Q MA for both y and Y ---
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

    % --- Drop NaN observations ---
    sel = ~isnan(sum([yH; YH; XC; XL; Z]));
    yH=yH(:,sel); YH=YH(:,sel); XL=XL(:,sel); XC=XC(:,sel); Z=Z(:,sel);
    T = size(yH, 2);

    % --- Partial out controls ---
    X  = [XC; XL; ones(1,T)];
    Nx = size(X, 1);
    yHo = yH - (yH/X)*X;
    YHo = YH - (YH/X)*X;
    Zo  = Z  - (Z/X)*X;

    % --- Reduced-form projections ---
    PYHo   = (YHo/Zo)*Zo;
    Sig_Zo = Zo*Zo'/T;
    irfsy  = (yHo/Zo) * Sig_Zo^0.5;
    irfsY  = (YHo/Zo) * Sig_Zo^0.5;

    % --- SP-IV estimate ---
    IH = eye(H);
    R  = kron(eye(K), IH(:));
    bThetay = reshape(irfsy, 1, H*Nz)';
    bThetaY = [];
    for k = 1:K
        bThetaY(:,k) = reshape(irfsY(1+(k-1)*H:k*H, :), H*Nz, 1);
    end
    b = bThetay' * bThetaY / (bThetaY' * bThetaY);

    % --- KLM confidence intervals (1D grid inversion) ---
    Ngrid = 201;
    bgrid = linspace(range(1), range(2), Ngrid);
    S_KLM   = zeros(1, Ngrid);
    OBJ_KLM = zeros(1, Ngrid);
    for c = 1:Ngrid
        uHo_c = yHo - kron(bgrid(c), eye(H)) * YHo;
        S_KLM(c)   = GetKstat(uHo_c, Zo, YHo, Nx, H, K);
        OBJ_KLM(c) = GetARObj(uHo_c, Zo, YHo);
    end

    % CUE estimate
    bmin = bgrid(OBJ_KLM == min(OBJ_KLM));
    fun1 = @(x) GetARObj(yHo - kron(x,eye(H))*YHo, Zo, YHo);
    options = optimoptions('fminunc', 'Display', 'off');
    [bmin,~,exitflag] = fminunc(fun1, bmin, options);

    % Find CI boundaries via fzero
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

    % --- Significance stars ---
    stars_KLM = sum(cil(clevel==[0.90 0.95 0.99])>0) + sum(ciu(clevel==[0.90 0.95 0.99])<0);

    RES.estimate   = b;
    RES.klm_lower  = cil;
    RES.klm_upper  = ciu;
    RES.stars_klm  = stars_KLM;
end


function RES = run_SPIV_ror_decomp(varsy, varsY, varsZ, varsZ_scale, ...
                                    Lcontrols, Ccontrols, range, sel_hors, p, clevel, DATA)
% SP-IV estimation for rate of return with two endogenous variables (K=2).
% Used for rows 4-7 of Table 2 (ND/D decomposition).
% Returns only GNRD (component 1) results.
%
% Both y and Y use nancumsum + 4Q MA since both are first-differences.
% Includes 2D KLM grid search with flood-fill for confidence intervals.

    K  = length(varsY);
    H  = length(sel_hors);
    sel_hors_all = 0:max(sel_hors);

    % --- Construct instruments ---
    if ~isempty(varsZ_scale)
        ins_names = strcat(varsZ, '_', varsZ_scale);
        ins_data = DATA(:,varsZ).Variables ./ lagmatrix(DATA(:,varsZ_scale).Variables, 1);
        D = array2timetable(ins_data, 'RowTimes', DATA.Time);
        D.Properties.VariableNames = ins_names;
        DATA = synchronize(D, DATA);
    else
        ins_names = varsZ;
    end
    Nz = length(ins_names);

    all_Lcontrols = unique([{varsy} Lcontrols]);

    % --- Extract data matrices ---
    y  = DATA(:, {varsy}).Variables';
    Y  = DATA(:, varsY).Variables';
    XL = lagmatrix(DATA(:, all_Lcontrols).Variables, 1:p)';
    XC = DATA(:, Ccontrols).Variables';
    Z  = DATA(:, ins_names).Variables';

    % --- Multi-horizon: nancumsum + 4Q MA for both y and Y ---
    yHt = (nancumsum(lagmatrix(y',-sel_hors_all),2)' + ...
           nancumsum(lagmatrix(y',-sel_hors_all+1),2)' + ...
           nancumsum(lagmatrix(y',-sel_hors_all+2),2)' + ...
           nancumsum(lagmatrix(y',-sel_hors_all+3),2)') / 4;
    yH = yHt(1+sel_hors, :);

    YH = [];
    for k = 1:K
        YHk = (nancumsum(lagmatrix(Y(k,:)',-sel_hors_all),2)' + ...
               nancumsum(lagmatrix(Y(k,:)',-sel_hors_all+1),2)' + ...
               nancumsum(lagmatrix(Y(k,:)',-sel_hors_all+2),2)' + ...
               nancumsum(lagmatrix(Y(k,:)',-sel_hors_all+3),2)') / 4;
        YH = [YH; YHk(1+sel_hors, :)];
    end

    % --- Drop NaN observations ---
    sel = ~isnan(sum([yH; YH; XC; XL; Z]));
    yH=yH(:,sel); YH=YH(:,sel); XL=XL(:,sel); XC=XC(:,sel); Z=Z(:,sel);
    T = size(yH, 2);

    % --- Partial out controls ---
    X  = [XC; XL; ones(1,T)];
    Nx = size(X, 1);
    yHo = yH - (yH/X)*X;
    YHo = YH - (YH/X)*X;
    Zo  = Z  - (Z/X)*X;

    % --- Reduced-form projections ---
    PYHo   = (YHo/Zo)*Zo;
    Sig_Zo = Zo*Zo'/T;
    irfsy  = (yHo/Zo) * Sig_Zo^0.5;
    irfsY  = (YHo/Zo) * Sig_Zo^0.5;

    % --- SP-IV estimate ---
    IH = eye(H);
    R  = kron(eye(K), IH(:));
    bThetay = reshape(irfsy, 1, H*Nz)';
    bThetaY = [];
    for k = 1:K
        bThetaY(:,k) = reshape(irfsY(1+(k-1)*H:k*H, :), H*Nz, 1);
    end
    b = bThetay' * bThetaY / (bThetaY' * bThetaY);

    % --- KLM confidence intervals (2D grid inversion) ---
    Ngrid = 101;
    d = NaN(Ngrid, K);
    for k = 1:K
        d(:,k) = linspace(range(k,1), range(k,2), Ngrid);
    end
    [bc1, bc2] = meshgrid(d(:,1), d(:,2));
    b1 = bc1';  b2 = bc2';

    S_KLM   = NaN(Ngrid, Ngrid);
    OBJ_KLM = NaN(Ngrid, Ngrid);
    for c1 = 1:Ngrid
        for c2 = 1:Ngrid
            beta = [b1(c1,c2) b2(c1,c2)];
            uHo_c = yHo - kron(beta, eye(H)) * YHo;
            S_KLM(c1,c2)   = GetKstat(uHo_c, Zo, YHo, Nx, H, K);
            OBJ_KLM(c1,c2) = GetARObj(uHo_c, Zo, YHo);
        end
    end

    % CUE estimate
    B1 = b1(:);  B2 = b2(:);
    idx_min = find(OBJ_KLM(:) == min(OBJ_KLM(:)));
    bmin0 = [B1(idx_min) B2(idx_min)];
    fun1 = @(x) GetARObj(yHo - kron(x,eye(H))*YHo, Zo, YHo);
    options = optimoptions('fminunc', 'Display', 'off');
    [bmin,~,exitflag] = fminunc(fun1, bmin0, options);
    if exitflag <= 0, bmin = NaN(1,K); end
    bcue = bmin;

    % Increase resolution via interpolation (10x)
    Ng = Ngrid * 10;
    d2 = NaN(Ng, K);
    for k = 1:K
        d2(:,k) = linspace(range(k,1), range(k,2), Ng);
    end
    [bbc1, bbc2] = meshgrid(d2(:,1), d2(:,2));
    S_KLM = interp2(bc1, bc2, S_KLM', bbc1, bbc2)';
    b1 = bbc1';  b2 = bbc2';
    B1 = b1(:);  B2 = b2(:);

    % Find grid position closest to CUE estimate
    dist = (B1-bcue(1)).^2 + (B2-bcue(2)).^2;
    minpoint = zeros(Ng^2, 1);
    minpoint(dist == min(dist)) = 1;
    minpoint = reshape(minpoint, Ng, Ng);
    [irow, icol] = find(minpoint);

    % Classify gridpoints by confidence level
    cval = chi2inv(clevel, K);
    test = zeros(size(S_KLM));
    for cl = 1:3
        test = test + (S_KLM > cval(cl));
    end
    test(end,end) = 3;

    % Flood-fill from CUE estimate to find connected confidence set
    test2 = 3*ones(size(test));
    test2(irow,icol) = 0;
    iter = 1;
    while iter < 3
        cleft = 0;
        irown = irow;
        while ~isempty(irown)
            for j = 1:length(irown)
                rup = 0;  flag_up = 0;
                while flag_up == 0
                    try
                        flag_up = (test(irown(j)-rup:irown(j), icol-cleft) == 3);
                        test2(irown(j)-rup:irown(j), icol-cleft) = 0;
                        rup = rup + 1;
                    catch
                        flag_up = 1;
                    end
                end
                rdown = 0;  flag_down = 0;
                while flag_down == 0
                    try
                        flag_down = (test(irown(j):irown(j)+rdown, icol-cleft) == 3);
                        test2(irown(j):irown(j)+rdown, icol-cleft) = 0;
                        rdown = rdown + 1;
                    catch
                        flag_down = 1;
                    end
                end
            end
            try
                if iter == 1
                    irown = find((test(:,icol-cleft-1)~=3) & (test2(:,icol-cleft)~=3));
                    cleft = cleft + 1;
                elseif iter == 2
                    irown = find((test(:,icol-cleft+1)~=3) & (test2(:,icol-cleft)~=3));
                    cleft = cleft - 1;
                end
                irown = irown((irown - lagmatrix(irown,1)) ~= 1);
            catch
                irown = [];
            end
        end
        iter = iter + 1;
    end
    testn = test;
    testn(test2 == 3) = 3;

    % Extract CI boundaries from connected set (GNRD = component 1 only)
    cil_klm = NaN(length(clevel), 1);
    ciu_klm = NaN(length(clevel), 1);
    cl = 0;
    for jj = 1:length(clevel)
        pts = (testn == cl);
        if any(pts(:))
            ciu_klm(jj) = max(B1(pts));
            cil_klm(jj) = min(B1(pts));
        end
        cl = cl + 1;
    end

    % --- Significance stars (GNRD component only) ---
    stars_KLM = sum(cil_klm(clevel==[0.90 0.95 0.99])>0) + ...
                sum(ciu_klm(clevel==[0.90 0.95 0.99])<0);

    RES.estimate   = b(1);
    RES.klm_lower  = cil_klm';
    RES.klm_upper  = ciu_klm';
    RES.stars_klm  = stars_KLM;
end


function AR_OBJ = GetARObj(uH, Z, YH)
% AR objective function for CUE estimation
    [~,T] = size(Z);
    PuH = (uH/Z)*Z;
    MuH = uH - PuH;
    AR_OBJ = T * trace(PuH*PuH' / (MuH*MuH'));
end


function Kstat = GetKstat(uH, Z, YH, Nx, H, K)
% K-statistic for KLM inference (robust to weak instruments)
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
