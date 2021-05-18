% experiment index, can be between 1 and 50
if ~exist('exp',  'var'), exp  =      1; end
% policy to run: 'single' for ENS, 'ucb' for UCB, 'ug' for UG, 'rep' for MF-ENS
if ~exist('name', 'var'), name =  'rep'; end
% dataset to use: 'ecfp', 'gpidaph', 'bmg'
if ~exist('data', 'var'), data = 'ecfp'; end

addpath(genpath('./'));
addpath(genpath('../active_learning'));
addpath(genpath('../active_search'));
addpath(genpath('../efficient_nonmyopic_active_search'));

%%% settings
theta     = 0.3          % simulated positive rate of the L fidelity
k         = 5            % the number of L queries for each H query
budget    = k * 300      % the total number of queries
lookahead = 1
verbose   = true;
data_dir  = '../data/';  % the directory containing the data

%%% problem setup
if contains(data, 'ecfp') || contains(data, 'gpidaph')
    data = sprintf('%s%d', data, exp);
end

data

[problem, labels_l, labels_h, weights, alpha, nns, sims] = load_mf_data( ...
    data, data_dir, theta);

if contains(data, 'citeseer') || contains(data, 'bmg')
    rng(exp);
else
    rng default;
end
pool      = and(labels_l == 1, labels_h == 1);
init_data = randsample((1:problem.num_points), 1, true, pool);

fprintf('false positive rate: %.4f\n', ...
        sum(and(labels_l == 1, labels_h == 2)) / sum(labels_l == 1));
fprintf('false negative rate: %.4f\n', ...
        sum(and(labels_l == 2, labels_h == 1)) / sum(labels_l == 2));

problem.budget    = budget;
problem.k         = k;
problem.time      = 0;
problem.running_h = [];
problem.verbose   = verbose;
problem.lookahead = lookahead;

other_weights = weights + speye(problem.num_points);
other_nns     = [(1:problem.num_points);       nns];
other_sims    = [ones(1, problem.num_points); sims];

probability_bound_h = get_mf_probability_bound( ...
    @mf_knn_probability_bound, weights, other_weights, ...
    nns', other_nns', sims', other_sims', alpha);
probability_bound_l = get_probability_bound_improved( ...
    @knn_probability_bound_improved, 4, weights, nns', sims', alpha);

model_h  = get_mf_model(@mf_knn_model, weights, other_weights, alpha);
model_l  = get_model(@knn_model_fast, weights, alpha);
selector = get_selector(@unlabeled_selector);
q_fn     = get_q(@get_q_mle, model_h);

%%% policy setup
switch name
case 'single'
    name   = 'single-fidelity ENS';
    policy = get_mf_policy(@single_fid_ens, model_h, model_l, weights, ...
                           other_weights, probability_bound_h);
case 'ucb'
    name     = 'ucb';
    beta_t_l = 0.01;
    beta_t_h = 0.001;
    l_policy = get_l_policy(@ucb_l, model_l, beta_t_l);
    h_policy = get_h_policy(@ucb_h, model_h, beta_t_h);
    policy   = get_mf_policy(@switch_l_h, l_policy, h_policy);
case 'ug'
    name     = 'uncertainty sampling on L, greedy on H';
    l_policy = get_l_policy(@uncertainty_sampling_l, model_h, model_l);
    h_policy = get_h_policy(@greedy_h, model_h);
    policy   = get_mf_policy(@switch_l_h, l_policy, h_policy);
case '1step'
    limit       = 1000;
    num_samples = 1000;
    name        = 'one-step ENS';
    if isfinite(limit)
        name = sprintf('%s limit %d', name, limit);
        if num_samples > 0
            name = sprintf('%s %d samples', name, num_samples);
        end
    end
    l_policy = get_l_policy(@one_step_ens_l_full, model_h, model_l, weights, other_weights, probability_bound_h, limit, num_samples);
    h_policy = get_h_policy(@one_step_ens_h, model_h, model_l, weights, other_weights, probability_bound_h, limit, num_samples);
    policy   = get_mf_policy(@switch_l_h, l_policy, h_policy);
otherwise  % rep
    limit             = 500;
    num_samples       = 500;
    num_label_samples = 2^k;
    name              = 'rep v2 shortened';
    if isfinite(limit)
        name = sprintf('%s limit %d', name, limit);
        if num_samples > 0
            name = sprintf('%s %d samples', name, num_samples);
        end
    end
    if num_label_samples < 2 ^ problem.k
        name = sprintf('%s %d label samples', name, num_label_samples);
    end
    l_policy = get_l_policy(@replace_shortened_l, model_h, model_l, weights, other_weights, probability_bound_h, probability_bound_l, limit, num_samples, num_label_samples);
    h_policy = get_h_policy(@replace_shortened_h, model_h, model_l, weights, other_weights, probability_bound_h, probability_bound_l, limit, num_samples, num_label_samples);
    policy   = get_mf_policy(@switch_l_h, l_policy, h_policy);
end

%%% run simulation
if problem.verbose
    name
end

rng(init_data);

message_prefix = sprintf('Exp %d:', exp);

train_ind_l = init_data;
train_ind_h = init_data;

observed_labels_l = labels_l(train_ind_l);
observed_labels_h = labels_h(train_ind_h);

[utilities, queries, queried_probs, computed, pruned1, pruned2] = ...
    mf_active_search(problem, train_ind_l, observed_labels_l, train_ind_h, ...
    observed_labels_h, labels_l, labels_h, selector, policy, q_fn, message_prefix);

%%% write results
time_str   = datetime(now, 'ConvertFrom', 'datenum');
result_dir = sprintf('./results/%d_%d_%d', ...
                     fix(budget / k), k, 100 - fix(theta * 100));

if contains(data, 'ecfp')
    result_dir = sprintf('%s/ecfp4/%s/', result_dir, name);
elseif contains(data, 'gpidaph')
    result_dir = sprintf('%s/gpidaph3/%s/', result_dir, name);
else
    result_dir = sprintf('%s/%s/%s/', result_dir, data, name);
end

if ~isdir(result_dir), mkdir(result_dir); end

writematrix(utilities, sprintf('%s/%s__utilities__%d__%d__%s.csv', ...
    result_dir, name, exp, init_data, time_str));
writematrix(queries, sprintf('%s/%s__queries__%d__%d__%s.csv', ...
    result_dir, name, exp, init_data, time_str));
writematrix(queried_probs, sprintf('%s/%s__queried_probs__%d__%d__%s.csv', ...
    result_dir, name, exp, init_data, time_str));

writematrix(computed, sprintf('%s/%s__computed__%d__%d__%s.csv', ...
    result_dir, name, exp, init_data, time_str));
writematrix(pruned1, sprintf('%s/%s__pruned1__%d__%d__%s.csv', ...
    result_dir, name, exp, init_data, time_str));
writematrix(pruned2, sprintf('%s/%s__pruned2__%d__%d__%s.csv', ...
    result_dir, name, exp, init_data, time_str));
