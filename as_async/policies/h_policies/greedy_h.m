% greedy sampling on H
function [cand_ind_h, prob_h, num_computed, num_pruned1, num_pruned2] = ...
    greedy_h(problem, train_ind_l, observed_labels_l, train_ind_h, ...
    observed_labels_h, unlabeled_ind_l, unlabeled_ind_h, model_h)

probs_h = model_h(problem, train_ind_l, observed_labels_l, ...
    train_ind_h, observed_labels_h, unlabeled_ind_h);
probs_h = probs_h(:, 1);

[prob_h, max_ind] = max(probs_h);
cand_ind_h         = unlabeled_ind_h(max_ind);

num_computed = numel(unlabeled_ind_h);
num_pruned1  = 0;
num_pruned2  = 0;
