% H-ENS on H, extended from ENS
function [cand_ind_h, prob_h, num_computed, num_pruned1, num_pruned2] = ...
    one_step_ens_h(problem, train_ind_l, observed_labels_l, train_ind_h, ...
    observed_labels_h, unlabeled_ind_l, unlabeled_ind_h, model_h, model_l, ...
    weights, other_weights, probability_bound_h, limit, num_samples)

if ~exist('limit',       'var'), limit       = Inf; end
if ~exist('num_samples', 'var'), num_samples = 0;   end

num_computed = 0;
num_pruned1  = 0;
num_pruned2  = 0;

num_points        = problem.num_points;
max_num_influence = problem.max_num_influence;
remain_budget_h   = fix((problem.budget - problem.time) / problem.k);

%% unlabeled points on H
unlabeled_ind_h = unlabeled_selector(problem, train_ind_h, []);
probs_h         = model_h(problem, train_ind_l, observed_labels_l, ...
                          train_ind_h, observed_labels_h, unlabeled_ind_h);
probs_h         = probs_h(:, 1);

[~, top_ind_h] = sort(probs_h, 'descend');
test_ind_h     = unlabeled_ind_h(top_ind_h);
num_tests_h    = numel(test_ind_h);

if problem.time > problem.budget - problem.k
    cand_ind_h   = test_ind_h(1);
    prob_h       = probs_h(top_ind_h(1));
    num_computed = numel(unlabeled_ind_h);
    return;
end

reverse_ind_h = zeros(num_points, 1);
reverse_ind_h(unlabeled_ind_h) = 1:num_tests_h;

weights(train_ind_h, :) = 0;

%% pruning
prob_upper_bound = probability_bound_h(problem, train_ind_l, observed_labels_l, ...
    train_ind_h, observed_labels_h, unlabeled_ind_l, unlabeled_ind_h, 0, 1, ...
    remain_budget_h);
future_utility_if_neg = sum(probs_h(top_ind_h(1:remain_budget_h)));

if max_num_influence >= remain_budget_h
    future_utility_if_pos = sum(prob_upper_bound(1:remain_budget_h));
else
    tmp_ind = top_ind_h(1:(remain_budget_h - max_num_influence));
    future_utility_if_pos = sum(probs_h(tmp_ind)) + ...
                            sum(prob_upper_bound(1:max_num_influence));
end

future_utility =      probs_h  * future_utility_if_pos + ...
                 (1 - probs_h) * future_utility_if_neg;

upper_bound_of_score = probs_h + future_utility;
upper_bound_of_score = upper_bound_of_score(top_ind_h);

pruned  = false(num_tests_h, 1);
score_h = -1;

for i = 1:num_tests_h
    if pruned(i)
        num_pruned1 = num_pruned1 + 1;
        continue;
    end
    if i > limit, break; end

    this_test_ind    = test_ind_h(i);
    fake_train_ind_h = [train_ind_h; this_test_ind];
    fake_test_ind_h  = find(weights(:, this_test_ind));
    success_prob     = probs_h(reverse_ind_h(this_test_ind));

    p = probs_h;
    p(reverse_ind_h(this_test_ind)) = 0;
    p(reverse_ind_h(fake_test_ind_h)) = 0;

    if isempty(fake_test_ind_h)
        top_bud_ind = top_ind_h(1:remain_budget_h);

        if find(top_bud_ind == reverse_ind_h(this_test_ind), 1)
            top_bud_ind = top_ind_h(1:(remain_budget_h + 1));
        end

        tmp_utility = probs_h(reverse_ind_h(this_test_ind)) + ...
                      sum(p(top_bud_ind));
    else
        fake_utilities = zeros(2, 1);
        tmp_pruned     = false;
        for fake_label = 1:2
            fake_observed_labels_h = [observed_labels_h; fake_label];
            fake_probs_h = model_h(problem, ...
                train_ind_l, observed_labels_l, ...
                fake_train_ind_h, fake_observed_labels_h, fake_test_ind_h);

            q = sort(fake_probs_h(:, 1), 'descend');

            fake_utilities(fake_label) = ...
                merge_sort(p, q, top_ind_h, remain_budget_h);

            if fake_label == 1 && (success_prob * (fake_utilities(1) + 1) + ...
                    (1 - success_prob) * future_utility_if_neg) <= score_h
                tmp_pruned = true;
                break;
            end
        end
        if tmp_pruned
            num_pruned2 = num_pruned2 + 1;
            continue;
        end

        tmp_utility  = success_prob + ...
            [success_prob, 1 - success_prob] * fake_utilities;
    end

    if tmp_utility > score_h
        score_h    = tmp_utility;
        cand_ind_h = this_test_ind;

        pruned(upper_bound_of_score <= score_h) = true;
    end

    num_computed = num_computed + 1;
end

num_pruned1 = num_pruned1 + sum(pruned(i:num_tests_h));

if (i < num_tests_h) && (num_samples > 0)
    candidates = (i:num_tests_h);
    candidates = candidates(~pruned(candidates));
    if num_samples < numel(candidates)
        candidates = sort(randsample(candidates, num_samples));
    end
    num_candidates = numel(candidates);

    for j = 1:num_candidates
        i = candidates(j);

        if pruned(i)
            num_pruned1 = num_pruned1 + 1;
            continue;
        end

        this_test_ind    = test_ind_h(i);
        fake_train_ind_h = [train_ind_h; this_test_ind];
        fake_test_ind_h  = find(weights(:, this_test_ind));

        p = probs_h;
        p(reverse_ind_h(this_test_ind)) = 0;
        p(reverse_ind_h(fake_test_ind_h)) = 0;

        if isempty(fake_test_ind_h)
            top_bud_ind = top_ind_h(1:remain_budget_h);

            if find(top_bud_ind == reverse_ind_h(this_test_ind), 1)
                top_bud_ind = top_ind_h(1:(remain_budget_h + 1));
            end

            tmp_utility = probs_h(reverse_ind_h(this_test_ind)) + ...
                          sum(p(top_bud_ind));
        else
            fake_utilities = zeros(2, 1);
            tmp_pruned     = false;
            for fake_label = 1:2
                fake_observed_labels_h = [observed_labels_h; fake_label];
                fake_probs_h = model_h(problem, ...
                    train_ind_l, observed_labels_l, ...
                    fake_train_ind_h, fake_observed_labels_h, fake_test_ind_h);

                q = sort(fake_probs_h(:, 1), 'descend');

                fake_utilities(fake_label) = ...
                    merge_sort(p, q, top_ind_h, remain_budget_h);

                if fake_label == 1 && (success_prob * (fake_utilities(1) + 1) + ...
                        (1 - success_prob) * future_utility_if_neg) <= score_h
                    tmp_pruned = true;
                    break;
                end
            end
            if tmp_pruned
                num_pruned2 = num_pruned2 + 1;
                continue;
            end

            tmp_utility = success_prob + ...
                [success_prob, 1 - success_prob] * fake_utilities;
        end

        if tmp_utility > score_h
            score_h    = tmp_utility;
            cand_ind_h = this_test_ind;

            pruned(upper_bound_of_score <= score_h) = true;
        end

        num_computed = num_computed + 1;
    end
end

prob_h = probs_h(reverse_ind_h(cand_ind_h));
