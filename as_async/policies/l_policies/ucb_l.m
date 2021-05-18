% MF-UCB on L
function [cand_ind_l, prob_l, num_computed, num_pruned1, num_pruned2] = ...
    ucb_l(problem, train_ind_l, observed_labels_l, train_ind_h, ...
    observed_labels_h, unlabeled_ind_l, unlabeled_ind_h, cand_ind_h, ...
    model_l, beta_t_l)

probs_l = model_l(problem, train_ind_l, observed_labels_l, unlabeled_ind_l);
probs_l = probs_l(:, 1);

ucb = probs_l + beta_t_l * sqrt(probs_l .* (1 - probs_l));  % UCB score

[score_l, max_ind] = max(ucb);
cand_ind_l         = unlabeled_ind_l(max_ind);
prob_l             = probs_l(max_ind);

num_computed = numel(unlabeled_ind_l);
num_pruned1  = 0;
num_pruned2  = 0;
