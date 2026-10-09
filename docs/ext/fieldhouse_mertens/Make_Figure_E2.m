%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Make_Figure_E2.m
%
% Generates Figure E.2: Simultaneous Weak-Instrument Robust Confidence
% Sets for ND/D R&D Elasticity Decomposition.
%
%   Figure_E2a.png - Intermediate eta (ln_tfp_util_adj2)
%   Figure_E2b.png - Low eta (ln_tfp_util_adj1)
%   Figure_E2c.png - High eta (ln_tfp_util_adj3)
%
% Each panel shows a 2D heatmap of the KLM test statistic over
% (phi_ND, phi_D) parameter space, with nested 90/95/99% confidence
% regions displayed via the pink colormap. SP-IV and CUE point
% estimates are overlaid as markers.
%
% Specification: ln_tfp_util_adj{1,2,3} on (dln_snK_GNRD, dln_snK_FDRD)
%   using z_NDEF instrument. Y variables are share-weighted first
%   differences of log R&D capital, cumulated via nancumsum + 4Q MA.
%
% Source: DATAQ timetable (must be in workspace)
% Requires: nancumsum.m, btrace.m, tightfig.m (in auxiliary_files/)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
warning off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1. SETTINGS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tfp_vars   = {'ln_tfp_util_adj2', 'ln_tfp_util_adj1', 'ln_tfp_util_adj3'};
fig_labels = {'a', 'b', 'c'};

varsZ      = {'z_NDEF'};
Lcontrols  = {'ln_cum_rALL_NDEF', 'ln_util', 'ln_tfp_util', ...
              'ln_rK_PRD_fernald', 'R_EW_COMB_TECH', ...
              'ln_rK_GRD', 'Military_news_scaled'};
horizons   = 3:4:59;
p          = 4;
clevel     = [0.90 0.95 0.99];
rang       = [-0.1 0.4; -0.5 0.75];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2. BUILD SHARE-WEIGHTED R&D CAPITAL GROWTH VARIABLES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

vars    = {'GRD', 'GNRD', 'FDRD'};
rk_vars = strcat('rK_', vars);
nK_vars = strcat('nK_', vars);

nK     = DATAQ(:, nK_vars).Variables;
shares = nK ./ nK(:,1);
shares = 0.5 * (shares + lagmatrix(shares, 1));

v = DATAQ(:, strcat('ln_', rk_vars)).Variables;
v = v - lagmatrix(v, 1);

D = array2timetable([shares .* v, shares], 'RowTimes', DATAQ.Time);
D.Properties.VariableNames = [strcat('dln_snK_', vars), strcat('shares_', vars)];
QQDATA = synchronize(DATAQ, D);

varsY = strcat('dln_snK_', vars(2:end));

K  = length(varsY);
H  = length(horizons);
Nz = length(varsZ);

if ~exist('figures/CS', 'dir'), mkdir('figures/CS'); end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 3. LOOP OVER TFP ADJUSTMENTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for spec = 1:3
    varsy = tfp_vars{spec};
   % fprintf('Figure E2%s: %s\n', fig_labels{spec}, varsy);

    all_Lcontrols = unique([{varsy} Lcontrols], 'stable');

    CS = compute_confidence_set(varsy, varsY, varsZ, all_Lcontrols, ...
        horizons, p, clevel, rang, QQDATA);

    % --- Plot ---
    f = figure;
    box off
    hold on
    imagesc(CS.bb2(1,:)', CS.bb1(:,1), CS.testn);
    colormap(pink);
    pp = gobjects(1, 2);
    pp(1) = plot(CS.b(1,2), CS.b(1,1), 'o', 'MarkerSize', 5, ...
        'MarkerFaceColor', [0.6350, 0.0780, 0.1840], ...
        'MarkerEdgeColor', [0.6350, 0.0780, 0.1840]);
    pp(2) = plot(CS.bcue(1,2), CS.bcue(1,1), 'o', 'MarkerSize', 5, ...
        'MarkerFaceColor', [0, 0.4470, 0.7410], ...
        'MarkerEdgeColor', [0, 0.4470, 0.7410]);
    xlim([-0.6 0.8])
    ylim([-0.1 0.3])
    xline(0)
    yline(0)
    ylabel('Nondefense R{\&}D Elasticity $\phi_{ND}$', 'Interpreter', 'Latex')
    xlabel('Defense R{\&}D Elasticity  $\phi_{D}$', 'Interpreter', 'Latex')
    l = legend(pp, 'SP-IV Point Estimate', 'SP-IV CUE Point Estimate');
    set(l, 'Location', 'NorthEast', 'Interpreter', 'Latex', 'Box', 'Off')
    xaxisproperties = get(gca, 'XAxis');
    xaxisproperties.TickLabelInterpreter = 'latex';
    yaxisproperties = get(gca, 'YAxis');
    yaxisproperties.TickLabelInterpreter = 'latex';
    tightfig;
    exportgraphics(gcf, sprintf('figures/Figure_E2%s.png', fig_labels{spec}), ...
        'Resolution', 300);
end

disp('Figure E.2 generated successfully.');
warning on

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% LOCAL FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function CS = compute_confidence_set(varsy, varsY, varsZ, all_Lcontrols, ...
    sel_hors, p, clevel, rang, DATA)
% Compute 2D KLM confidence set for (phi_ND, phi_D) decomposition.

    K  = length(varsY);
    H  = length(sel_hors);
    Nz = length(varsZ);
    sel_hors_all = 0:max(sel_hors);

    y  = DATA(:, {varsy}).Variables';
    Y  = DATA(:, varsY).Variables';
    XL = lagmatrix(DATA(:, all_Lcontrols).Variables, 1:p)';
    Z  = DATA(:, varsZ).Variables';

    IH = eye(H);
    R  = kron(eye(K), IH(:));

    % Multi-horizon outcomes: y = 4Q MA of levels, Y = nancumsum + 4Q MA
    yHt = (lagmatrix(y',-sel_hors_all)' + lagmatrix(y',-sel_hors_all+1)' + ...
           lagmatrix(y',-sel_hors_all+2)' + lagmatrix(y',-sel_hors_all+3)') / 4;
    yH  = yHt(1+sel_hors, :);

    YH = [];
    for k = 1:K
        YHk = (nancumsum(lagmatrix(Y(k,:)',-sel_hors_all),2)' + ...
               nancumsum(lagmatrix(Y(k,:)',-sel_hors_all+1),2)' + ...
               nancumsum(lagmatrix(Y(k,:)',-sel_hors_all+2),2)' + ...
               nancumsum(lagmatrix(Y(k,:)',-sel_hors_all+3),2)') / 4;
        YH = [YH; YHk(1+sel_hors, :)];
    end

    % Drop NaN observations
    sel = ~isnan(sum([yH; YH; XL; Z]));
    yH = yH(:,sel); YH = YH(:,sel); XL = XL(:,sel); Z = Z(:,sel);
    T  = size(yH, 2);

    X  = [XL; ones(1,T)];
    Nx = size(X, 1);

    % Project out controls
    yHo = yH - (yH/X)*X;
    YHo = YH - (YH/X)*X;
    Zo  = Z  - (Z/X)*X;

    % First-stage projections
    PYHo   = (YHo/Zo)*Zo;
    Sig_Zo = Zo*Zo'/T;
    irfsy  = (yHo/Zo) * Sig_Zo^0.5;
    irfsY  = (YHo/Zo) * Sig_Zo^0.5;

    % SP-IV point estimate
    bThetay = reshape(irfsy, 1, H*Nz)';
    bThetaY = [];
    for k = 1:K
        bThetaY(:,k) = reshape(irfsY(1+(k-1)*H:k*H, :), H*Nz, 1);
    end
    b = bThetay' * bThetaY / (bThetaY' * bThetaY);

    % --- 2D Grid of KLM test statistics (201 x 201) ---
    Ng0 = 201;
    d = NaN(Ng0, K);
    for k = 1:K
        d(:,k) = linspace(rang(k,1), rang(k,2), Ng0);
    end
    [bc1, bc2] = meshgrid(d(:,1), d(:,2));
    b1 = bc1'; b2 = bc2';

    S_KLM   = NaN(Ng0, Ng0);
    OBJ_KLM = NaN(Ng0, Ng0);
    for c1 = 1:Ng0
        for c2 = 1:Ng0
            beta = [b1(c1,c2) b2(c1,c2)];
            uHo = yHo - kron(beta, eye(H)) * YHo;
            S_KLM(c1,c2)   = GetKstat(uHo, Zo, YHo, Nx, H, K);
            OBJ_KLM(c1,c2) = GetARObj(uHo, Zo, YHo);
        end
    end

    % CUE estimator
    B1 = b1(:); B2 = b2(:);
    idx_min = find(OBJ_KLM(:) == min(OBJ_KLM(:)));
    bmin0 = [B1(idx_min) B2(idx_min)];
    fun1 = @(x) GetARObj(yHo - kron(x, eye(H)) * YHo, Zo, YHo);
    options = optimoptions('fminunc', 'Display', 'off');
    [bmin, ~, exitflag] = fminunc(fun1, bmin0, options);
    if exitflag > 0
        bcue = bmin;
    else
        bcue = NaN(1, K);
    end

    % --- Interpolate to 10x resolution ---
    Ng = Ng0 * 10;
    d2 = NaN(Ng, K);
    for k = 1:K
        d2(:,k) = linspace(rang(k,1), rang(k,2), Ng);
    end
    [bbc1, bbc2] = meshgrid(d2(:,1), d2(:,2));
    b1 = bbc1'; b2 = bbc2';
    S_KLM = interp2(bc1, bc2, S_KLM', bbc1, bbc2)';

    bb1 = b1; bb2 = b2;
    B1 = bb1(:); B2 = bb2(:);

    % Find grid point closest to CUE
    dist = (B1 - bcue(1)).^2 + (B2 - bcue(2)).^2;
    minpoint = zeros(Ng^2, 1);
    minpoint(dist == min(dist)) = 1;
    minpoint = reshape(minpoint, Ng, Ng);
    [irow, icol] = find(minpoint);

    % --- Classify confidence levels ---
    cval = chi2inv(clevel, K);
    test = zeros(size(S_KLM));
    for cl = 1:3
        test = test + (S_KLM > cval(cl));
    end
    test(end, end) = 3;

    % --- Boundary detection algorithm ---
    test2 = 3 * ones(size(test));
    test2(irow, icol) = 0;
    iter = 1;
    while iter < 3
        cleft = 0;
        irown = irow;
        while ~isempty(irown)
            for j = 1:length(irown)
                % Expand upward
                rup = 0; flag_up = 0;
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
                rdown = 0; flag_down = 0;
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
                    irown = find((test(:,icol-cleft-1) ~= 3) & (test2(:,icol-cleft) ~= 3));
                    cleft = cleft + 1;
                elseif iter == 2
                    irown = find((test(:,icol-cleft+1) ~= 3) & (test2(:,icol-cleft) ~= 3));
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

    % --- Output ---
    CS.bb1   = bb1;
    CS.bb2   = bb2;
    CS.testn = testn;
    CS.test  = test;
    CS.b     = b;
    CS.bcue  = bcue;
end


function AR_OBJ = GetARObj(uH, Z, YH)
    [~, T] = size(Z);
    PuH = (uH/Z)*Z;
    MuH = uH - PuH;
    AR_OBJ = T * trace(PuH*PuH' / (MuH*MuH'));
end


function Kstat = GetKstat(uH, Z, YH, Nx, H, K)
    [Nz, T] = size(Z);
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
