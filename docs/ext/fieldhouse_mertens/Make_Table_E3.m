%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Make_Table_E3.m
%
% Generates Table E.3: Estimates of Production Function Elasticities of
% Total R&D Capital
%
% This table reports SP-IV elasticity estimates where the endogenous
% variable is total R&D capital (government + private, from Fernald).
%
% Rows 1-2: "Total (G+P)" — single endogenous variable ln_rK_TRD_fernald,
%           1D KLM grid search. Row 1 uses Exo ND, Row 2 uses Exo D.
% Rows 3-4: "G/P" decomposition — two endogenous variables
%           (dln_snK_GRD, dln_snK_PRD_fernald) with share-weighted growth,
%           2D KLM grid search (101x101 + 10x interpolation + flood-fill).
%           Row 3 uses Exo ND, Row 4 uses Exo ND/D.
%
% Output:  tables/results_Table_E3.tex  (LaTeX table body, 4 rows)
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

% Share-weighted capital growth for G/P decomposition (rows 3-4).
% TRD_fernald = GRD + PRD_fernald (government + private R&D capital).
decomp_vars = {'TRD_fernald', 'GRD', 'PRD_fernald'};
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

clevel   = [0.90 0.95 0.99];
horizons = 3:4:59;
p_lags   = 4;

tfp_vars = {'ln_tfp_util_adj1', 'ln_tfp_util_adj2', 'ln_tfp_util_adj3'};

Lcontrols_ND = {'ln_cum_rALL_NDEF', 'ln_util', 'ln_tfp_util', ...
                'ln_rK_PRD_fernald', 'R_EW_COMB_TECH', ...
                'ln_rK_GRD', 'Military_news_scaled'};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 3. RUN SP-IV FOR ALL SPECIFICATIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Preallocate: 4 rows x 3 TFP variants
ND_est   = NaN(4,3);  ND_ci  = NaN(4,3,2);  ND_stars = NaN(4,3);
D_est    = NaN(4,3);  D_ci   = NaN(4,3,2);  D_stars  = NaN(4,3);

% --- Row 1: Total (G+P), Exo ND ---
% fprintf('Row 1: Total (G+P), Exo ND\n');
for t = 1:3
    RES = run_SPIV_total(tfp_vars{t}, {'ln_rK_TRD_fernald'}, {'z_NDEF'}, {}, ...
        Lcontrols_ND, {}, [-2 2], horizons, p_lags, clevel, QQDATA);
    ND_est(1,t)=RES.estimate; ND_ci(1,t,:)=[RES.klm_lower(2) RES.klm_upper(2)]; ND_stars(1,t)=RES.stars_klm;
end

% --- Row 2: Total (G+P), Exo D ---
% fprintf('Row 2: Total (G+P), Exo D\n');
for t = 1:3
    RES = run_SPIV_total(tfp_vars{t}, {'ln_rK_TRD_fernald'}, {'z_NSEC'}, {}, ...
        Lcontrols_ND, {}, [-2 2], horizons, p_lags, clevel, QQDATA);
    ND_est(2,t)=RES.estimate; ND_ci(2,t,:)=[RES.klm_lower(2) RES.klm_upper(2)]; ND_stars(2,t)=RES.stars_klm;
end

% --- Row 3: G/P Decomposition, Exo ND ---
% fprintf('Row 3: G/P Decomp, Exo ND\n');
for t = 1:3
    [R1,R2] = run_SPIV_decomp(tfp_vars{t}, {'dln_snK_GRD','dln_snK_PRD_fernald'}, ...
        {'z_NDEF'}, {}, Lcontrols_ND, {}, [-2 2; -2 2], ...
        horizons, p_lags, clevel, QQDATA);
    ND_est(3,t)=R1.estimate; ND_ci(3,t,:)=[R1.klm_lower(2) R1.klm_upper(2)]; ND_stars(3,t)=R1.stars_klm;
    D_est(3,t)=R2.estimate;  D_ci(3,t,:)=[R2.klm_lower(2) R2.klm_upper(2)];  D_stars(3,t)=R2.stars_klm;
end

% --- Row 4: G/P Decomposition, Exo ND/D ---
% fprintf('Row 4: G/P Decomp, Exo ND/D\n');
for t = 1:3
    [R1,R2] = run_SPIV_decomp(tfp_vars{t}, {'dln_snK_GRD','dln_snK_PRD_fernald'}, ...
        {'z_NDEF','z_NSEC'}, {}, Lcontrols_ND, {}, [-2 2; -2 2], ...
        horizons, p_lags, clevel, QQDATA);
    ND_est(4,t)=R1.estimate; ND_ci(4,t,:)=[R1.klm_lower(2) R1.klm_upper(2)]; ND_stars(4,t)=R1.stars_klm;
    D_est(4,t)=R2.estimate;  D_ci(4,t,:)=[R2.klm_lower(2) R2.klm_upper(2)];  D_stars(4,t)=R2.stars_klm;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 4. FORMAT AND WRITE LaTeX TABLE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

label1 = {'Total (G+P)', 'Total (G+P)', 'G/P', 'G/P'};
label2 = {'Exo ND', 'Exo D', 'Exo ND', 'Exo ND/D'};

fileID = fopen('tables/results_Table_E3.tex', 'w');
for row = 1:4
    fprintf(fileID, '\\footnotesize{$[%6.0f]$}&', row);
    fprintf(fileID, '\\footnotesize{%s}', label1{row});
    fprintf(fileID, '&');
    fprintf(fileID, '\\footnotesize{%s}', label2{row});

    % ND/G elasticity, moderate adjustment (column 4)
    if ~isnan(ND_est(row,2))
        write_cell(fileID, ND_est(row,2), squeeze(ND_ci(row,2,:)), ND_stars(row,2));
    else
        fprintf(fileID, '&  ');
    end

    % D/P elasticity, moderate adjustment (column 5)
    if ~isnan(D_est(row,2))
        write_cell(fileID, D_est(row,2), squeeze(D_ci(row,2,:)), D_stars(row,2));
    else
        fprintf(fileID, '&  ');
    end

    fprintf(fileID, '&  ');  % blank separator

    % ND/G elasticity, low adjustment (column 7) — stars from moderate
    if ~isnan(ND_est(row,1))
        write_cell(fileID, ND_est(row,1), squeeze(ND_ci(row,1,:)), ND_stars(row,2));
    else
        fprintf(fileID, '&  ');
    end

    fprintf(fileID, '&  ');  % blank separator

    % ND/G elasticity, high adjustment (column 9) — own stars
    if ~isnan(ND_est(row,3))
        write_cell(fileID, ND_est(row,3), squeeze(ND_ci(row,3,:)), ND_stars(row,3));
    else
        fprintf(fileID, '&  ');
    end

    fprintf(fileID, '\\\\ \n');
end
fclose(fileID);

disp('Table E.3 generated successfully: tables/results_Table_E3.tex');
warning on

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% LOCAL FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function write_cell(fid, est, ci, nstars)
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
% 4-quarter moving average of levels for both y and Y.

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

    PYHo   = (YHo/Zo)*Zo;
    Sig_Zo = Zo*Zo'/T;
    irfsy  = (yHo/Zo) * Sig_Zo^0.5;
    irfsY  = (YHo/Zo) * Sig_Zo^0.5;

    IH = eye(H);
    R  = kron(eye(K), IH(:));
    vH     = YHo - PYHo;
    SigvH  = vH*vH' / (size(vH,2)-Nz-Nx);
    Omega  = R'*kron(SigvH,eye(H))*R;
    gmin   = min(eig(Omega^-0.5 * R'*kron(PYHo*PYHo',eye(H))*R * Omega^-0.5 / Nz));

    bThetay = reshape(irfsy, 1, H*Nz)';
    bThetaY = [];
    for k = 1:K
        bThetaY(:,k) = reshape(irfsY(1+(k-1)*H:k*H, :), H*Nz, 1);
    end
    b = bThetay' * bThetaY / (bThetaY' * bThetaY);

    uHo = yHo - kron(b, eye(H)) * YHo;
    SiguHo = uHo*uHo' / (T-K-Nx);
    Sig_beta = (R'*kron(PYHo*PYHo',eye(H))*R) \ ...
               R'*kron(PYHo*PYHo',SiguHo)*R / ...
               (R'*kron(PYHo*PYHo',eye(H))*R);
    for jj = 1:length(clevel)
        ciuw(jj) = b + tinv(1-(1-clevel(jj))/2, T-K-Nx)*sqrt(Sig_beta);
        cilw(jj) = b - tinv(1-(1-clevel(jj))/2, T-K-Nx)*sqrt(Sig_beta);
    end

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
    [bmin,~,exitflag] = fminunc(fun1, bmin, options);

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

    stars_KLM = sum(cil(clevel==[0.90 0.95 0.99])>0) + sum(ciu(clevel==[0.90 0.95 0.99])<0);

    RES.estimate   = b;
    RES.klm_lower  = cil;
    RES.klm_upper  = ciu;
    RES.stars_klm  = stars_KLM;
end


function [RES1, RES2] = run_SPIV_decomp(varsy, varsY, varsZ, varsZ_scale, ...
                                         Lcontrols, Ccontrols, range, sel_hors, p, clevel, DATA)
% SP-IV estimation with two endogenous variables (K=2).
% y (TFP level): 4Q moving average of levels.
% Y (share-weighted capital growth): nancumsum + 4Q moving average.
% 2D grid search (101x101) + 10x interpolation + flood-fill.

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

    % y: 4Q MA of levels
    yHt = (lagmatrix(y',-sel_hors_all)' + lagmatrix(y',-sel_hors_all+1)' + ...
           lagmatrix(y',-sel_hors_all+2)' + lagmatrix(y',-sel_hors_all+3)') / 4;
    yH = yHt(1+sel_hors, :);

    % Y: nancumsum + 4Q MA
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

    % Wald CIs
    uHo = yHo - kron(b, eye(H)) * YHo;
    SiguHo = uHo*uHo' / (T-K-Nx);
    Sig_beta = (R'*kron(PYHo*PYHo',eye(H))*R) \ ...
               R'*kron(PYHo*PYHo',SiguHo)*R / ...
               (R'*kron(PYHo*PYHo',eye(H))*R);
    for jj = 1:length(clevel)
        ciuw(jj,:) = b + tinv(1-(1-clevel(jj))/2, T-K-Nx)*sqrt(diag(Sig_beta)');
        cilw(jj,:) = b - tinv(1-(1-clevel(jj))/2, T-K-Nx)*sqrt(diag(Sig_beta)');
    end

    % 2D KLM grid (101x101)
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

    % 10x interpolation
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

    % Flood-fill from CUE estimate
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

    % Extract CI boundaries
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

    % Significance stars
    starsK = [sum(cil_klm(clevel==[0.90 0.95 0.99],1)>0) + sum(ciu_klm(clevel==[0.90 0.95 0.99],1)<0), ...
              sum(cil_klm(clevel==[0.90 0.95 0.99],2)>0) + sum(ciu_klm(clevel==[0.90 0.95 0.99],2)<0)];

    RES1.estimate   = b(1);
    RES1.klm_lower  = cil_klm(:,1)';
    RES1.klm_upper  = ciu_klm(:,1)';
    RES1.stars_klm  = starsK(1);

    RES2.estimate   = b(2);
    RES2.klm_lower  = cil_klm(:,2)';
    RES2.klm_upper  = ciu_klm(:,2)';
    RES2.stars_klm  = starsK(2);
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
