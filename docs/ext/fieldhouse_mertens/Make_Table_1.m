%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Make_Table_1.m
%
% Generates Table 1: Estimates of Production Function Elasticities of
% Government R&D Capital
%
% This script estimates production function elasticities using System
% Projections on Instrument Variables (SP-IV). The SP-IV estimator
% projects TFP and R&D capital onto narrative appropriations shock
% instruments at multiple horizons, then regresses TFP projections on
% capital projections. KLM confidence intervals provide robust inference.
%
% (SP-IV = System Projections on Instrumental Variables)
%
% The table reports SP-IV elasticity estimates for three infrastructure
% TFP adjustments (low, moderate, high) with 95% KLM confidence intervals.
%
% Output:  tables/results_Table_1.tex  (LaTeX table body, 9 rows)
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

% Share-weighted capital growth rates for ND/D decomposition (rows 6-9).
% GRD = GNRD + FDRD (government nondefense + federal defense R&D capital).
% Growth is weighted by Divisia shares: average of current and lagged
% nominal shares, times the log-change in real capital.
decomp_vars = {'GRD', 'GNRD', 'FDRD'};
nK = DATAQ(:, strcat('nK_', decomp_vars)).Variables;
shares = nK ./ nK(:,1);
shares = 0.5 * (shares + lagmatrix(shares, 1));
dlnrK  = DATAQ(:, strcat('ln_rK_', decomp_vars)).Variables;
dlnrK  = dlnrK - lagmatrix(dlnrK, 1);
D = array2timetable([shares .* dlnrK, shares], 'RowTimes', DATAQ.Time);
D.Properties.VariableNames = [strcat('dln_snK_', decomp_vars), ...
                               strcat('shares_', decomp_vars)];
QQDATA = synchronize(DATAQ, D);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2. ESTIMATION SETTINGS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clevel   = [0.90 0.95 0.99];   % confidence levels
horizons = 3:4:59;              % every 4th quarter from Q4
p_lags   = 4;                   % number of lags

% TFP variants: low, moderate, high infrastructure adjustment
tfp_vars = {'ln_tfp_util_adj1', 'ln_tfp_util_adj2', 'ln_tfp_util_adj3'};

% Baseline lagged control variables
Lcontrols_ND = {'ln_cum_rALL_NDEF', 'ln_util', 'ln_tfp_util', ...
                'ln_rK_PRD_fernald', 'R_EW_COMB_TECH', ...
                'ln_rK_GRD', 'Military_news_scaled'};
Lcontrols_D  = {'ln_cum_rALL_NSEC', 'ln_util', 'ln_tfp_util', ...
                'ln_rK_PRD_fernald', 'R_EW_COMB_TECH', ...
                'ln_rK_GRD', 'Military_news_scaled'};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 3. RUN SP-IV FOR ALL SPECIFICATIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Preallocate result arrays: 9 rows x 3 TFP variants
ND_est   = NaN(9,3);  ND_ci  = NaN(9,3,2);  ND_stars = NaN(9,3);
D_est    = NaN(9,3);  D_ci   = NaN(9,3,2);  D_stars  = NaN(9,3);

% --- Row 1: Total GRD, Exo ND ---
% fprintf('Row 1: Total GRD, Exo ND\n');
for t = 1:3
    RES = run_SPIV_total(tfp_vars{t}, {'ln_rK_GRD'}, {'z_NDEF'}, {}, ...
        Lcontrols_ND, {}, [-1 2], horizons, p_lags, clevel, QQDATA);
    ND_est(1,t)=RES.estimate; ND_ci(1,t,:)=[RES.klm_lower(2) RES.klm_upper(2)]; ND_stars(1,t)=RES.stars_klm;
end

% --- Row 2: Total GRD, Exo ND No NASA ---
% fprintf('Row 2: Total GRD, Exo ND No NASA\n');
for t = 1:3
    RES = run_SPIV_total(tfp_vars{t}, {'ln_rK_GRD'}, {'z_NDEF_exspace'}, {}, ...
        Lcontrols_ND, {'z_SPACE'}, [-1 2], horizons, p_lags, clevel, QQDATA);
    ND_est(2,t)=RES.estimate; ND_ci(2,t,:)=[RES.klm_lower(2) RES.klm_upper(2)]; ND_stars(2,t)=RES.stars_klm;
end

% --- Row 3: Total GRD, All ND ---
% fprintf('Row 3: Total GRD, All ND\n');
for t = 1:3
    RES = run_SPIV_total(tfp_vars{t}, {'ln_rK_GRD'}, {'z_allNDEF'}, {}, ...
        Lcontrols_ND, {}, [-2 2], horizons, p_lags, clevel, QQDATA);
    ND_est(3,t)=RES.estimate; ND_ci(3,t,:)=[RES.klm_lower(2) RES.klm_upper(2)]; ND_stars(3,t)=RES.stars_klm;
end

% --- Row 4: Total GRD, Exo D ---
% fprintf('Row 4: Total GRD, Exo D\n');
for t = 1:3
    RES = run_SPIV_total(tfp_vars{t}, {'ln_rK_GRD'}, {'z_NSEC'}, {}, ...
        Lcontrols_D, {}, [-2 2], horizons, p_lags, clevel, QQDATA);
    D_est(4,t)=RES.estimate; D_ci(4,t,:)=[RES.klm_lower(2) RES.klm_upper(2)]; D_stars(4,t)=RES.stars_klm;
end

% --- Row 5: Total GRD, All D ---
% fprintf('Row 5: Total GRD, All D\n');
for t = 1:3
    RES = run_SPIV_total(tfp_vars{t}, {'ln_rK_GRD'}, {'z_allNSEC'}, {}, ...
        Lcontrols_D, {}, [-2 2], horizons, p_lags, clevel, QQDATA);
    D_est(5,t)=RES.estimate; D_ci(5,t,:)=[RES.klm_lower(2) RES.klm_upper(2)]; D_stars(5,t)=RES.stars_klm;
end

% --- Row 6: ND/D Decomposition, Exo ND ---
% fprintf('Row 6: ND/D Decomp, Exo ND\n');
for t = 1:3
    [R1,R2] = run_SPIV_decomp(tfp_vars{t}, {'dln_snK_GNRD','dln_snK_FDRD'}, ...
        {'z_NDEF'}, {}, Lcontrols_ND, {}, [-0.1 0.4; -0.5 0.75], ...
        horizons, p_lags, clevel, QQDATA);
    ND_est(6,t)=R1.estimate; ND_ci(6,t,:)=[R1.klm_lower(2) R1.klm_upper(2)]; ND_stars(6,t)=R1.stars_klm;
    D_est(6,t)=R2.estimate;  D_ci(6,t,:)=[R2.klm_lower(2) R2.klm_upper(2)];  D_stars(6,t)=R2.stars_klm;
end

% --- Row 7: ND/D Decomposition, Exo ND/D ---
% fprintf('Row 7: ND/D Decomp, Exo ND/D\n');
for t = 1:3
    [R1,R2] = run_SPIV_decomp(tfp_vars{t}, {'dln_snK_GNRD','dln_snK_FDRD'}, ...
        {'rEXO_NDEF','rEXO_NSEC'}, {'rK_FNRD','rK_FDRD'}, ...
        Lcontrols_ND, {}, [-2 2; -2 2], horizons, p_lags, clevel, QQDATA);
    ND_est(7,t)=R1.estimate; ND_ci(7,t,:)=[R1.klm_lower(2) R1.klm_upper(2)]; ND_stars(7,t)=R1.stars_klm;
    D_est(7,t)=R2.estimate;  D_ci(7,t,:)=[R2.klm_lower(2) R2.klm_upper(2)];  D_stars(7,t)=R2.stars_klm;
end

% --- Row 8: ND/D Decomposition, Exo ND No NASA ---
% fprintf('Row 8: ND/D Decomp, Exo ND No NASA\n');
for t = 1:3
    [R1,R2] = run_SPIV_decomp(tfp_vars{t}, {'dln_snK_GNRD','dln_snK_FDRD'}, ...
        {'z_NDEF_exspace'}, {}, Lcontrols_ND, {'z_SPACE'}, ...
        [-2 2; -2 2], horizons, p_lags, clevel, QQDATA);
    ND_est(8,t)=R1.estimate; ND_ci(8,t,:)=[R1.klm_lower(2) R1.klm_upper(2)]; ND_stars(8,t)=R1.stars_klm;
    D_est(8,t)=R2.estimate;  D_ci(8,t,:)=[R2.klm_lower(2) R2.klm_upper(2)];  D_stars(8,t)=R2.stars_klm;
end

% --- Row 9: ND/D Decomposition, All ND ---
% fprintf('Row 9: ND/D Decomp, All ND\n');
for t = 1:3
    [R1,R2] = run_SPIV_decomp(tfp_vars{t}, {'dln_snK_GNRD','dln_snK_FDRD'}, ...
        {'z_allNDEF'}, {}, Lcontrols_ND, {}, [-0.1 0.4; -0.5 0.75], ...
        horizons, p_lags, clevel, QQDATA);
    ND_est(9,t)=R1.estimate; ND_ci(9,t,:)=[R1.klm_lower(2) R1.klm_upper(2)]; ND_stars(9,t)=R1.stars_klm;
    D_est(9,t)=R2.estimate;  D_ci(9,t,:)=[R2.klm_lower(2) R2.klm_upper(2)];  D_stars(9,t)=R2.stars_klm;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 4. FORMAT AND WRITE LaTeX TABLE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

label1 = {'Total','Total','Total','Total','Total','ND/D','ND/D','ND/D','ND/D'};
label2 = {'Exo ND','Exo ND, No NASA','All ND','Exo D','All D', ...
           'Exo ND','Exo ND/D','Exo ND, No NASA','All ND'};

fileID = fopen('tables/results_Table_1.tex', 'w');
for row = 1:9
    % Row number, decomposition label, instrument label
    fprintf(fileID, '\\footnotesize{$[%6.0f]$}&', row);
    fprintf(fileID, '\\footnotesize{%s}', label1{row});
    fprintf(fileID, '&');
    fprintf(fileID, '\\footnotesize{%s}', label2{row});

    % ND elasticity, moderate adjustment (column 4)
    if ~isnan(ND_est(row,2))
        write_cell(fileID, ND_est(row,2), squeeze(ND_ci(row,2,:)), ND_stars(row,2));
    else
        fprintf(fileID, '&  ');
    end

    % D elasticity, moderate adjustment (column 5)
    if ~isnan(D_est(row,2))
        write_cell(fileID, D_est(row,2), squeeze(D_ci(row,2,:)), D_stars(row,2));
    else
        fprintf(fileID, '&  ');
    end

    fprintf(fileID, '&  ');  % blank separator

    % ND elasticity, low adjustment (column 7) — stars from moderate
    if ~isnan(ND_est(row,1))
        write_cell(fileID, ND_est(row,1), squeeze(ND_ci(row,1,:)), ND_stars(row,2));
    else
        fprintf(fileID, '&  ');
    end

    fprintf(fileID, '&  ');  % blank separator

    % ND elasticity, high adjustment (column 9) — own stars
    if ~isnan(ND_est(row,3))
        write_cell(fileID, ND_est(row,3), squeeze(ND_ci(row,3,:)), ND_stars(row,3));
    else
        fprintf(fileID, '&  ');
    end

    fprintf(fileID, '\\\\ \n');
end
fclose(fileID);

disp('Table 1 generated successfully: tables/results_Table_1.tex');
warning on

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% LOCAL FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function write_cell(fid, est, ci, nstars)
% Write one table cell: estimate with 95% CI underneath, significance stars
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


function RES = run_SPIV_total(varsy, varsY, varsZ, varsZ_scale, ...
                               Lcontrols, Ccontrols, range, sel_hors, p, clevel, DATA)
% SP-IV estimation with single endogenous variable (K=1).
% Used for rows 1-5 of Table 1 (Total GRD specifications).
%
% The estimator projects TFP (varsy) and capital (varsY) onto instruments
% (varsZ) at the specified horizons using 4-quarter moving averages,
% then regresses TFP projections on capital projections.

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

    % --- Lagged controls: add TFP variable (include_lags_of_y = 1) ---
    all_Lcontrols = [{varsy} Lcontrols];

    % --- Extract data matrices ---
    y  = DATA(:, {varsy}).Variables';
    Y  = DATA(:, varsY).Variables';
    XL = lagmatrix(DATA(:, all_Lcontrols).Variables, 1:p)';
    XC = DATA(:, Ccontrols).Variables';
    Z  = DATA(:, ins_names).Variables';

    % --- Multi-horizon outcomes (4-quarter moving average) ---
    yHt = (lagmatrix(y',-sel_hors_all)' + lagmatrix(y',-sel_hors_all+1)' + ...
           lagmatrix(y',-sel_hors_all+2)' + lagmatrix(y',-sel_hors_all+3)') / 4;
    yH  = yHt(1+sel_hors, :);

    YH = [];
    for k = 1:K
        YHk = (lagmatrix(Y(k,:)',-sel_hors_all)' + lagmatrix(Y(k,:)',-sel_hors_all+1)' + ...
               lagmatrix(Y(k,:)',-sel_hors_all+2)' + lagmatrix(Y(k,:)',-sel_hors_all+3)') / 4;
        YH = [YH; YHk(1+sel_hors, :)];
    end

    % --- Drop NaN observations ---
    sel = ~isnan(sum([yH; YH; XC; XL; Z]));
    yH=yH(:,sel); YH=YH(:,sel); XL=XL(:,sel); XC=XC(:,sel); Z=Z(:,sel);
    dates = DATA.Time; dates = dates(sel);
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

    % --- First-stage test ---
    IH = eye(H);
    R  = kron(eye(K), IH(:));
    vH     = YHo - PYHo;
    SigvH  = vH*vH' / (size(vH,2)-Nz-Nx);
    Omega  = R'*kron(SigvH,eye(H))*R;
    gmin   = min(eig(Omega^-0.5 * R'*kron(PYHo*PYHo',eye(H))*R * Omega^-0.5 / Nz));

    % --- SP-IV estimate (second stage) ---
    bThetay = reshape(irfsy, 1, H*Nz)';
    bThetaY = [];
    for k = 1:K
        bThetaY(:,k) = reshape(irfsY(1+(k-1)*H:k*H, :), H*Nz, 1);
    end
    b = bThetay' * bThetaY / (bThetaY' * bThetaY);

    % --- Wald confidence intervals ---
    uHo = yHo - kron(b, eye(H)) * YHo;
    SiguHo = uHo*uHo' / (T-K-Nx);
    Sig_beta = (R'*kron(PYHo*PYHo',eye(H))*R) \ ...
               R'*kron(PYHo*PYHo',SiguHo)*R / ...
               (R'*kron(PYHo*PYHo',eye(H))*R);
    for jj = 1:length(clevel)
        ciuw(jj) = b + tinv(1-(1-clevel(jj))/2, T-K-Nx)*sqrt(Sig_beta);
        cilw(jj) = b - tinv(1-(1-clevel(jj))/2, T-K-Nx)*sqrt(Sig_beta);
    end

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

    % CUE estimate (minimize AR objective)
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

        % Lower boundary
        goi_l = goi(goi < gp_est);
        if ~isempty(goi_l)
            bg_l = goi_l(knnsearch(goi_l, gp_est));
        else
            bg_l = 1;
        end
        % Upper boundary
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
    stars_W   = sum(cilw(clevel==[0.90 0.95 0.99])>0) + sum(ciuw(clevel==[0.90 0.95 0.99])<0);
    stars_KLM = sum(cil(clevel==[0.90 0.95 0.99])>0)  + sum(ciu(clevel==[0.90 0.95 0.99])<0);

    RES.estimate   = b;
    RES.klm_lower  = cil;
    RES.klm_upper  = ciu;
    RES.wald_lower = cilw;
    RES.wald_upper = ciuw;
    RES.stars_klm  = stars_KLM;
    RES.stars_wald = stars_W;
    RES.gmin       = gmin;
end


function [RES1, RES2] = run_SPIV_decomp(varsy, varsY, varsZ, varsZ_scale, ...
                                         Lcontrols, Ccontrols, range, sel_hors, p, clevel, DATA)
% SP-IV estimation with two endogenous variables (K=2).
% Used for rows 6-9 of Table 1 (ND/D decomposition).
%
% varsY = {'dln_snK_GNRD', 'dln_snK_FDRD'} — share-weighted growth rates
% These are cumulated across horizons via nancumsum before projection.
% RES1 = results for first component (GNRD / nondefense)
% RES2 = results for second component (FDRD / defense)

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

    % --- Lagged controls: add TFP variable, apply unique ---
    all_Lcontrols = unique([{varsy} Lcontrols]);

    % --- Extract data matrices ---
    y  = DATA(:, {varsy}).Variables';
    Y  = DATA(:, varsY).Variables';
    XL = lagmatrix(DATA(:, all_Lcontrols).Variables, 1:p)';
    XC = DATA(:, Ccontrols).Variables';
    Z  = DATA(:, ins_names).Variables';

    % --- Multi-horizon outcomes ---
    % TFP (y): 4-quarter moving average of levels
    yHt = (lagmatrix(y',-sel_hors_all)' + lagmatrix(y',-sel_hors_all+1)' + ...
           lagmatrix(y',-sel_hors_all+2)' + lagmatrix(y',-sel_hors_all+3)') / 4;
    yH = yHt(1+sel_hors, :);

    % Capital growth (Y): cumulate growth rates (nancumsum), then 4Q MA
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
    dates = DATA.Time; dates = dates(sel);
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

    % --- First-stage test ---
    IH = eye(H);
    R  = kron(eye(K), IH(:));
    vH     = YHo - PYHo;
    SigvH  = vH*vH' / (size(vH,2)-Nz-Nx);
    Omega  = R'*kron(SigvH,eye(H))*R;
    gmin   = min(eig(Omega^-0.5 * R'*kron(PYHo*PYHo',eye(H))*R * Omega^-0.5 / Nz));

    % --- SP-IV estimate (second stage) ---
    bThetay = reshape(irfsy, 1, H*Nz)';
    bThetaY = [];
    for k = 1:K
        bThetaY(:,k) = reshape(irfsY(1+(k-1)*H:k*H, :), H*Nz, 1);
    end
    b = bThetay' * bThetaY / (bThetaY' * bThetaY);  % 1 x K vector

    % --- Wald confidence intervals ---
    uHo = yHo - kron(b, eye(H)) * YHo;
    SiguHo = uHo*uHo' / (T-K-Nx);
    Sig_beta = (R'*kron(PYHo*PYHo',eye(H))*R) \ ...
               R'*kron(PYHo*PYHo',SiguHo)*R / ...
               (R'*kron(PYHo*PYHo',eye(H))*R);
    for jj = 1:length(clevel)
        ciuw(jj,:) = b + tinv(1-(1-clevel(jj))/2, T-K-Nx)*sqrt(diag(Sig_beta)');
        cilw(jj,:) = b - tinv(1-(1-clevel(jj))/2, T-K-Nx)*sqrt(diag(Sig_beta)');
    end

    % --- KLM confidence intervals (2D grid inversion) ---
    Ngrid = 201;
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
                % Expand upward
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
                % Expand downward
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

    % Extract CI boundaries from connected set
    cil_klm = NaN(length(clevel), K);
    ciu_klm = NaN(length(clevel), K);
    cl = 0;
    for jj = 1:length(clevel)
        pts = (testn == cl);
        if any(pts(:))
            ciu_klm(jj,:) = [max(B1(pts)) max(B2(pts))];
            cil_klm(jj,:) = [min(B1(pts)) min(B2(pts))];
        end
        cl = cl + 1;
    end

    % --- Significance stars for each component ---
    starsW = [sum(cilw(clevel==[0.90 0.95 0.99],1)>0) + sum(ciuw(clevel==[0.90 0.95 0.99],1)<0), ...
              sum(cilw(clevel==[0.90 0.95 0.99],2)>0) + sum(ciuw(clevel==[0.90 0.95 0.99],2)<0)];
    starsK = [sum(cil_klm(clevel==[0.90 0.95 0.99],1)>0) + sum(ciu_klm(clevel==[0.90 0.95 0.99],1)<0), ...
              sum(cil_klm(clevel==[0.90 0.95 0.99],2)>0) + sum(ciu_klm(clevel==[0.90 0.95 0.99],2)<0)];

    % Component 1 (GNRD / nondefense)
    RES1.estimate   = b(1);
    RES1.klm_lower  = cil_klm(:,1)';
    RES1.klm_upper  = ciu_klm(:,1)';
    RES1.wald_lower = cilw(:,1)';
    RES1.wald_upper = ciuw(:,1)';
    RES1.stars_klm  = starsK(1);
    RES1.stars_wald = starsW(1);
    RES1.gmin       = gmin;

    % Component 2 (FDRD / defense)
    RES2.estimate   = b(2);
    RES2.klm_lower  = cil_klm(:,2)';
    RES2.klm_upper  = ciu_klm(:,2)';
    RES2.wald_lower = cilw(:,2)';
    RES2.wald_upper = ciuw(:,2)';
    RES2.stars_klm  = starsK(2);
    RES2.stars_wald = starsW(2);
    RES2.gmin       = gmin;
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
