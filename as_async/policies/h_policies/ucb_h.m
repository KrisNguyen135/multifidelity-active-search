% MF-UCB on H
function [cand_ind_h, prob_h, num_computed, num_pruned1, num_pruned2] = ...
    ucb_h(problem, train_ind_l, observed_labels_l, train_ind_h, ...
    observed_labels_h, unlabeled_ind_l, unlabeled_ind_h, model_h, beta_t_h)

probs_h = model_h(problem, train_ind_l, observed_labels_l, ...
    train_ind_h, observed_labels_h, unlabeled_ind_h);
probs_h = probs_h(:, 1);

ucb = probs_h + beta_t_h * sqrt(probs_h .* (1 - probs_h));  % UCB score

[score_h, max_ind] = max(ucb);
cand_ind_h         = unlabeled_ind_h(max_ind);
prob_h             = probs_h(max_ind);

num_computed = numel(unlabeled_ind_h);
num_pruned1  = 0;
num_pruned2  = 0;
