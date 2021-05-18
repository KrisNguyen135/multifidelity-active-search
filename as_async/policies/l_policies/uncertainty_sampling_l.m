% uncertainty sampling on L
function [cand_ind_l, prob_l, num_computed, num_pruned1, num_pruned2] = ...
    uncertainty_sampling_l(problem, train_ind_l, observed_labels_l, ...
    train_ind_h, observed_labels_h, unlabeled_ind_l, unlabeled_ind_h, ...
    cand_ind_h, model_h, model_l)

if ~isempty(cand_ind_h)
    unlabeled_ind_h = unlabeled_ind_h(find(unlabeled_ind_h ~= cand_ind_h));
end
unlabeled_ind_both = intersect(unlabeled_ind_l, unlabeled_ind_h);

probs_h = model_h(problem, train_ind_l, observed_labels_l, ...
    train_ind_h, observed_labels_h, unlabeled_ind_both);
probs_h = probs_h(:, 1);

[score_l, min_ind] = min(abs(probs_h - 0.5));  % most uncertainty, closest to 0.5
cand_ind_l         = unlabeled_ind_both(min_ind);
prob_l             = model_l(problem, train_ind_l, observed_labels_l, cand_ind_l);
prob_l             = prob_l(:, 1);

num_computed = numel(unlabeled_ind_l);
num_pruned1  = 0;
num_pruned2  = 0;
