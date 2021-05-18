function [cand_ind_h, cand_ind_l, prob_h, prob_l, num_computed_h, num_pruned1_h, ...
    num_pruned2_h, num_computed_l, num_pruned1_l, num_pruned2_l] = ...
    single_fid_ens(problem, train_ind_l, observed_labels_l, train_ind_h, ...
    observed_labels_h, unlabeled_ind_l, unlabeled_ind_h, return_both, ...
    model_h, model_l, weights, other_weights, probability_bound_h)

if return_both
    %% if querying on H, run H-ENS as it is equivalent
    [cand_ind_h, prob_h, num_computed_h, num_pruned1_h, num_pruned2_h] = ...
        one_step_ens_h(problem, [], [], train_ind_h, observed_labels_h, [], ...
        unlabeled_ind_h, model_h, model_l, weights, other_weights, ...
        probability_bound_h);
else
    %% otherwise, return empty and NaN
    cand_ind_h = [];
    prob_h     = NaN;

    num_computed_h = 0;
    num_pruned1_h  = 0;
    num_pruned2_h  = 0;
end

cand_ind_l = [];
prob_l     = NaN;

num_computed_l = 0;
num_pruned1_l  = 0;
num_pruned2_l  = 0;
