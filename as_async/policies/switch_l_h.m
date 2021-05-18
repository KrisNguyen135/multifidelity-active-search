% helper function to combine an H policy and an L policy
% see run_sim.m to see how policies are combined
function [cand_ind_h, cand_ind_l, score_h, score_l, num_computed_h, num_pruned1_h, ...
    num_pruned2_h, num_computed_l, num_pruned1_l, num_pruned2_l] = ...
    switch_l_h(problem, train_ind_l, observed_labels_l, train_ind_h, ...
    observed_labels_h, unlabeled_ind_l, unlabeled_ind_h, return_both, ...
    l_policy, h_policy)

if return_both
    %% when an H query and an L query are needed
    [cand_ind_h, score_h, num_computed_h, num_pruned1_h, num_pruned2_h] = ...
        h_policy(problem, train_ind_l, observed_labels_l, train_ind_h, ...
        observed_labels_h, unlabeled_ind_l, unlabeled_ind_h);

    if problem.time == problem.budget - problem.k + 1
        cand_ind_l = [];
        score_l    = NaN;

        num_computed_l = 0;
        num_pruned1_l  = 0;
        num_pruned2_l  = 0;
        return;
    end
else
    %% when only an L query is needed
    cand_ind_h = problem.running_h;
    score_h    = NaN;

    num_computed_h = 0;
    num_pruned1_h  = 0;
    num_pruned2_h  = 0;
end

[cand_ind_l, score_l, num_computed_l, num_pruned1_l, num_pruned2_l] = l_policy(...
    problem, train_ind_l, observed_labels_l, train_ind_h, observed_labels_h, ...
    unlabeled_ind_l, unlabeled_ind_h, cand_ind_h);
