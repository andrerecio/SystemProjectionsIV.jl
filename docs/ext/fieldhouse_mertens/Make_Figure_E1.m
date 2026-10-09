%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Make_Figure_E1.m
%
% Generates Figure E.1: Illustration of the SP-IV Estimator
%
%   Figure_E1a.png - TFP responses and SP-IV fitted values for three
%                    infrastructure elasticity assumptions
%   Figure_E1b.png - Government R&D capital response to instrument shock
%
% For each infrastructure elasticity assumption (eta), TFP is adjusted
% differently for public infrastructure's contribution. The SP-IV
% estimator regresses the TFP response (irfsy) on the capital response
% (irfsY) across horizons. Solid lines show the actual TFP responses;
% dashed lines show the SP-IV fitted values (beta * irfsY).
%
% Specification: ln_tfp_util_adj{1,2,3,4} on ln_rK_GRD using z_NDEF
%   adj1 = low eta (0.065), adj2 = intermediate, adj3 = high (0.12),
%   adj4 = very high (0.39)
%
% Source: DATAQ timetable (must be in workspace)
% Requires: tightfig.m (in auxiliary_files/)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
warning off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1. SETTINGS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

varsY     = {'ln_rK_GRD'};
varsZ     = {'z_NDEF'};
Lcontrols = {'ln_cum_rALL_NDEF', 'ln_util', 'ln_tfp_util', ...
             'ln_rK_PRD_fernald', 'R_EW_COMB_TECH', ...
             'ln_rK_GRD', 'Military_news_scaled'};
horizons  = 3:4:59;
p         = 4;

tfp_vars = {'ln_tfp_util_adj1', 'ln_tfp_util_adj2', ...
            'ln_tfp_util_adj3', 'ln_tfp_util_adj4'};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2. COMPUTE SP-IV PROJECTIONS FOR EACH TFP ADJUSTMENT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

irfsy_all = [];
spiv_all  = zeros(4, 1);

for i = 1:4
    varsy = tfp_vars{i};
    all_Lcontrols = [{varsy} Lcontrols];

    [irfsy_i, irfsY_i, spiv_i] = get_spiv_projections(varsy, varsY, ...
        varsZ, all_Lcontrols, horizons, p, DATAQ);

    irfsy_all(:, i) = irfsy_i * 100;
    spiv_all(i) = spiv_i;

    if i == 1
        irfsY = irfsY_i * 100;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 3. FIGURE E.1a: TFP RESPONSES AND SP-IV FITTED VALUES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure
hold on
p_handles = gobjects(1, 3);

% Solid lines: actual TFP responses (adj1, adj3, adj4)
p_handles(1) = plot(horizons, irfsy_all(:,1), 'Color', [0, 0.4470, 0.7410]);
p_handles(2) = plot(horizons, irfsy_all(:,3), 'Color', [0.8500, 0.3250, 0.0980]);
p_handles(3) = plot(horizons, irfsy_all(:,4), 'Color', [0.9290, 0.6940, 0.1250]);

% Dashed lines: SP-IV fitted values (spiv_estimate * irfsY)
plot(horizons, spiv_all(1) * irfsY(:,1), '--', 'Color', [0, 0.4470, 0.7410])
plot(horizons, spiv_all(3) * irfsY(:,1), '--', 'Color', [0.8500, 0.3250, 0.0980])
plot(horizons, spiv_all(4) * irfsY(:,1), '--', 'Color', [0.9290, 0.6940, 0.1250])

xlabel('horizon (quarters)', 'Interpreter', 'Latex')
ylabel('Percent', 'Interpreter', 'Latex')
yline(0);
ylim([-0.10 0.35])

l = legend(p_handles, ...
    {'Low Public Infrastructure Elasticity, $\eta=0.065$', ...
     'High Public Infrastructure Elasticity, $\eta=0.12$', ...
     'Very High Public Infrastructure Elasticity, $\eta=0.39$'});
set(l, 'Location', 'Best', 'Interpreter', 'Latex', 'Box', 'Off')

xaxisproperties = get(gca, 'XAxis');
xaxisproperties.TickLabelInterpreter = 'latex';
yaxisproperties = get(gca, 'YAxis');
yaxisproperties.TickLabelInterpreter = 'latex';
tightfig;
exportgraphics(gcf, 'figures/Figure_E1a.png', 'Resolution', 300);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 4. FIGURE E.1b: GOVERNMENT R&D CAPITAL RESPONSE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure
hold on
plot(horizons, irfsY)
xlabel('horizon (quarters)', 'Interpreter', 'Latex')
ylabel('Percent', 'Interpreter', 'Latex')
yline(0);
ylim([0 2.7])

xaxisproperties = get(gca, 'XAxis');
xaxisproperties.TickLabelInterpreter = 'latex';
yaxisproperties = get(gca, 'YAxis');
yaxisproperties.TickLabelInterpreter = 'latex';
tightfig;
exportgraphics(gcf, 'figures/Figure_E1b.png', 'Resolution', 300);

disp('Figure E.1 generated successfully.');
warning on

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% LOCAL FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [irfsy, irfsY, spiv_estimate] = get_spiv_projections(varsy, ...
    varsY, varsZ, all_Lcontrols, sel_hors, p, DATA)
% Compute SP-IV projections: irfsy, irfsY (instrument IRFs) and the
% SP-IV point estimate (K=1, 4Q MA of levels).

    K  = length(varsY);
    H  = length(sel_hors);
    Nz = length(varsZ);
    sel_hors_all = 0:max(sel_hors);

    y  = DATA(:, {varsy}).Variables';
    Y  = DATA(:, varsY).Variables';
    XL = lagmatrix(DATA(:, all_Lcontrols).Variables, 1:p)';
    Z  = DATA(:, varsZ).Variables';

    % Multi-horizon outcomes: 4-quarter moving average of levels
    yHt = (lagmatrix(y',-sel_hors_all)' + lagmatrix(y',-sel_hors_all+1)' + ...
           lagmatrix(y',-sel_hors_all+2)' + lagmatrix(y',-sel_hors_all+3)') / 4;
    yH  = yHt(1+sel_hors, :);

    YH = [];
    for k = 1:K
        YHk = (lagmatrix(Y(k,:)',-sel_hors_all)' + lagmatrix(Y(k,:)',-sel_hors_all+1)' + ...
               lagmatrix(Y(k,:)',-sel_hors_all+2)' + lagmatrix(Y(k,:)',-sel_hors_all+3)') / 4;
        YH = [YH; YHk(1+sel_hors, :)];
    end

    % Drop NaN observations
    sel = ~isnan(sum([yH; YH; XL; Z]));
    yH = yH(:,sel); YH = YH(:,sel); XL = XL(:,sel); Z = Z(:,sel);
    T = size(yH, 2);

    % Project out controls
    X   = [XL; ones(1,T)];
    yHo = yH - (yH/X)*X;
    YHo = YH - (YH/X)*X;
    Zo  = Z  - (Z/X)*X;

    % Instrument IRFs
    Sig_Zo = Zo*Zo'/T;
    irfsy  = (yHo/Zo) * Sig_Zo^0.5;
    irfsY  = (YHo/Zo) * Sig_Zo^0.5;

    % SP-IV estimate
    IH = eye(H);
    bThetay = reshape(irfsy, 1, H*Nz)';
    bThetaY = [];
    for k = 1:K
        bThetaY(:,k) = reshape(irfsY(1+(k-1)*H:k*H, :), H*Nz, 1);
    end
    spiv_estimate = bThetay' * bThetaY / (bThetaY' * bThetaY);
end
