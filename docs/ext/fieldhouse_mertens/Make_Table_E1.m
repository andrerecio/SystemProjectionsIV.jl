%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Make_Table_E1.m
%
% Generates Appendix Table E.1: SP-IV Elasticity Estimates with Wald
% Inference
%
% This script estimates production function elasticities using System
% Projections on Instrument Variables (SP-IV). The SP-IV estimator
% projects TFP and R&D capital onto narrative appropriations shock
% instruments at multiple horizons, then regresses TFP projections on
% capital projections. Wald confidence intervals are computed from the
% asymptotic variance of the estimator.
%
% (SP-IV = System Projections on Instrumental Variables)
%
% The table reports SP-IV elasticity estimates for three infrastructure
% TFP adjustments (low, moderate, high) with 95% Wald confidence intervals.
%
% Output:  tables/results_Table_E1.tex  (LaTeX table body, 9 rows)
%
% Source:  DATAQ timetable (must be in workspace)
% Requires: nancumsum.m (in auxiliary_files/)
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
fprintf('Row 1: Total GRD, Exo ND\n');
for t = 1:3
    RES = run_SPIV_total(tfp_vars{t}, {'ln_rK_GRD'}, {'z_NDEF'}, {}, ...
        Lcontrols_ND, {}, horizons, p_lags, clevel, QQDATA);
    ND_est(1,t)=RES.estimate; ND_ci(1,t,:)=[RES.wald_lower(2) RES.wald_upper(2)]; ND_stars(1,t)=RES.stars_wald;
end

% --- Row 2: Total GRD, Exo ND No NASA ---
fprintf('Row 2: Total GRD, Exo ND No NASA\n');
for t = 1:3
    RES = run_SPIV_total(tfp_vars{t}, {'ln_rK_GRD'}, {'z_NDEF_exspace'}, {}, ...
        Lcontrols_ND, {'z_SPACE'}, horizons, p_lags, clevel, QQDATA);
    ND_est(2,t)=RES.estimate; ND_ci(2,t,:)=[RES.wald_lower(2) RES.wald_upper(2)]; ND_stars(2,t)=RES.stars_wald;
end

% --- Row 3: Total GRD, All ND ---
fprintf('Row 3: Total GRD, All ND\n');
for t = 1:3
    RES = run_SPIV_total(tfp_vars{t}, {'ln_rK_GRD'}, {'z_allNDEF'}, {}, ...
        Lcontrols_ND, {}, horizons, p_lags, clevel, QQDATA);
    ND_est(3,t)=RES.estimate; ND_ci(3,t,:)=[RES.wald_lower(2) RES.wald_upper(2)]; ND_stars(3,t)=RES.stars_wald;
end

% --- Row 4: Total GRD, Exo D ---
fprintf('Row 4: Total GRD, Exo D\n');
for t = 1:3
    RES = run_SPIV_total(tfp_vars{t}, {'ln_rK_GRD'}, {'z_NSEC'}, {}, ...
        Lcontrols_D, {}, horizons, p_lags, clevel, QQDATA);
    D_est(4,t)=RES.estimate; D_ci(4,t,:)=[RES.wald_lower(2) RES.wald_upper(2)]; D_stars(4,t)=RES.stars_wald;
end

% --- Row 5: Total GRD, All D ---
fprintf('Row 5: Total GRD, All D\n');
for t = 1:3
    RES = run_SPIV_total(tfp_vars{t}, {'ln_rK_GRD'}, {'z_allNSEC'}, {}, ...
        Lcontrols_D, {}, horizons, p_lags, clevel, QQDATA);
    D_est(5,t)=RES.estimate; D_ci(5,t,:)=[RES.wald_lower(2) RES.wald_upper(2)]; D_stars(5,t)=RES.stars_wald;
end

% --- Row 6: ND/D Decomposition, Exo ND ---
fprintf('Row 6: ND/D Decomp, Exo ND\n');
for t = 1:3
    [R1,R2] = run_SPIV_decomp(tfp_vars{t}, {'dln_snK_GNRD','dln_snK_FDRD'}, ...
        {'z_NDEF'}, {}, Lcontrols_ND, {}, horizons, p_lags, clevel, QQDATA);
    ND_est(6,t)=R1.estimate; ND_ci(6,t,:)=[R1.wald_lower(2) R1.wald_upper(2)]; ND_stars(6,t)=R1.stars_wald;
    D_est(6,t)=R2.estimate;  D_ci(6,t,:)=[R2.wald_lower(2) R2.wald_upper(2)];  D_stars(6,t)=R2.stars_wald;
end

% --- Row 7: ND/D Decomposition, Exo ND/D ---
fprintf('Row 7: ND/D Decomp, Exo ND/D\n');
for t = 1:3
    [R1,R2] = run_SPIV_decomp(tfp_vars{t}, {'dln_snK_GNRD','dln_snK_FDRD'}, ...
        {'rEXO_NDEF','rEXO_NSEC'}, {'rK_FNRD','rK_FDRD'}, ...
        Lcontrols_ND, {}, horizons, p_lags, clevel, QQDATA);
    ND_est(7,t)=R1.estimate; ND_ci(7,t,:)=[R1.wald_lower(2) R1.wald_upper(2)]; ND_stars(7,t)=R1.stars_wald;
    D_est(7,t)=R2.estimate;  D_ci(7,t,:)=[R2.wald_lower(2) R2.wald_upper(2)];  D_stars(7,t)=R2.stars_wald;
end

% --- Row 8: ND/D Decomposition, Exo ND No NASA ---
fprintf('Row 8: ND/D Decomp, Exo ND No NASA\n');
for t = 1:3
    [R1,R2] = run_SPIV_decomp(tfp_vars{t}, {'dln_snK_GNRD','dln_snK_FDRD'}, ...
        {'z_NDEF_exspace'}, {}, Lcontrols_ND, {'z_SPACE'}, ...
        horizons, p_lags, clevel, QQDATA);
    ND_est(8,t)=R1.estimate; ND_ci(8,t,:)=[R1.wald_lower(2) R1.wald_upper(2)]; ND_stars(8,t)=R1.stars_wald;
    D_est(8,t)=R2.estimate;  D_ci(8,t,:)=[R2.wald_lower(2) R2.wald_upper(2)];  D_stars(8,t)=R2.stars_wald;
end

% --- Row 9: ND/D Decomposition, All ND ---
fprintf('Row 9: ND/D Decomp, All ND\n');
for t = 1:3
    [R1,R2] = run_SPIV_decomp(tfp_vars{t}, {'dln_snK_GNRD','dln_snK_FDRD'}, ...
        {'z_allNDEF'}, {}, Lcontrols_ND, {}, horizons, p_lags, clevel, QQDATA);
    ND_est(9,t)=R1.estimate; ND_ci(9,t,:)=[R1.wald_lower(2) R1.wald_upper(2)]; ND_stars(9,t)=R1.stars_wald;
    D_est(9,t)=R2.estimate;  D_ci(9,t,:)=[R2.wald_lower(2) R2.wald_upper(2)];  D_stars(9,t)=R2.stars_wald;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 4. FORMAT AND WRITE LaTeX TABLE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

label1 = {'Total','Total','Total','Total','Total','ND/D','ND/D','ND/D','ND/D'};
label2 = {'Exo ND','Exo ND, No NASA','All ND','Exo D','All D', ...
           'Exo ND','Exo ND/D','Exo ND, No NASA','All ND'};

fileID = fopen('tables/results_Table_E1.tex', 'w');
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

disp('Table E.1 generated successfully: tables/results_Table_E1.tex');
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
                               Lcontrols, Ccontrols, sel_hors, p, clevel, DATA)
% SP-IV estimation with single endogenous variable (K=1).
% Used for rows 1-5 of Table E.1 (Total GRD specifications).
%
% Computes the SP-IV point estimate and Wald confidence intervals.
% No grid search is needed for Wald inference.

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

    % --- SP-IV estimate (second stage) ---
    IH = eye(H);
    R  = kron(eye(K), IH(:));
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
    ciuw = NaN(1, length(clevel));
    cilw = NaN(1, length(clevel));
    for jj = 1:length(clevel)
        ciuw(jj) = b + tinv(1-(1-clevel(jj))/2, T-K-Nx)*sqrt(Sig_beta);
        cilw(jj) = b - tinv(1-(1-clevel(jj))/2, T-K-Nx)*sqrt(Sig_beta);
    end

    % --- Significance stars ---
    stars_W = sum(cilw(clevel==[0.90 0.95 0.99])>0) + sum(ciuw(clevel==[0.90 0.95 0.99])<0);

    RES.estimate   = b;
    RES.wald_lower = cilw;
    RES.wald_upper = ciuw;
    RES.stars_wald = stars_W;
end


function [RES1, RES2] = run_SPIV_decomp(varsy, varsY, varsZ, varsZ_scale, ...
                                         Lcontrols, Ccontrols, sel_hors, p, clevel, DATA)
% SP-IV estimation with two endogenous variables (K=2).
% Used for rows 6-9 of Table E.1 (ND/D decomposition).
%
% varsY = {'dln_snK_GNRD', 'dln_snK_FDRD'} — share-weighted growth rates
% These are cumulated across horizons via nancumsum before projection.
% Computes the SP-IV point estimates and Wald confidence intervals.
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

    % --- SP-IV estimate (second stage) ---
    IH = eye(H);
    R  = kron(eye(K), IH(:));
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
    ciuw = NaN(length(clevel), K);
    cilw = NaN(length(clevel), K);
    for jj = 1:length(clevel)
        ciuw(jj,:) = b + tinv(1-(1-clevel(jj))/2, T-K-Nx)*sqrt(diag(Sig_beta)');
        cilw(jj,:) = b - tinv(1-(1-clevel(jj))/2, T-K-Nx)*sqrt(diag(Sig_beta)');
    end

    % --- Significance stars for each component ---
    starsW = [sum(cilw(clevel==[0.90 0.95 0.99],1)>0) + sum(ciuw(clevel==[0.90 0.95 0.99],1)<0), ...
              sum(cilw(clevel==[0.90 0.95 0.99],2)>0) + sum(ciuw(clevel==[0.90 0.95 0.99],2)<0)];

    % Component 1 (GNRD / nondefense)
    RES1.estimate   = b(1);
    RES1.wald_lower = cilw(:,1)';
    RES1.wald_upper = ciuw(:,1)';
    RES1.stars_wald = starsW(1);

    % Component 2 (FDRD / defense)
    RES2.estimate   = b(2);
    RES2.wald_lower = cilw(:,2)';
    RES2.wald_upper = ciuw(:,2)';
    RES2.stars_wald = starsW(2);
end
