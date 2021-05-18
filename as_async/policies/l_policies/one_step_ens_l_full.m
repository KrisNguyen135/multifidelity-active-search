% H-ENS on L
function [cand_ind_l, prob_l, num_computed, num_pruned1, num_pruned2] = ...
    one_step_ens_l_full(problem, train_ind_l, observed_labels_l, ...
    train_ind_h, observed_labels_h, unlabeled_ind_l, unlabeled_ind_h, ...
    cand_ind_h, model_h, model_l, weights, other_weights, ...
    probability_bound_h, limit, num_samples)

function [tmp_utility, tmp_pruned] = ...
    one_step_ens_l_full_compute_score(i, this_test_ind)

    tmp_utility      = -1;
    tmp_pruned       = false;
    fake_train_ind_l = [train_ind_l; this_test_ind];
    fake_test_ind_h  = find(other_weights(:, this_test_ind));

    p_if_neg = probs_h_if_neg;
    p_if_neg(reverse_ind_h(fake_test_ind_h)) = 0;
    p_if_pos = probs_h_if_pos;
    p_if_pos(reverse_ind_h(fake_test_ind_h)) = 0;

    if isempty(fake_test_ind_h)
        tmp_utility_if_neg = sum(p_if_neg(top_k_ind_h_if_neg(1:remain_budget_h)));
        tmp_utility_if_pos = sum(p_if_pos(top_k_ind_h_if_pos(1:remain_budget_h)));

        tmp_utility = pending_success_prob * tmp_utility_if_pos + ...
            (1 - pending_success_prob) * tmp_utility_if_neg;
    else
        success_prob = probs_l(reverse_ind_l(this_test_ind));

        %% if cand_ind_h is negative
        fake_utilities = zeros(2, 1);
        for fake_label = 1:2
            fake_observed_labels_l = [observed_labels_l; fake_label];
            fake_probs_h = model_h(problem, fake_train_ind_l, ...
                fake_observed_labels_l, fake_train_ind_h, ...
                fake_observed_labels_h_if_neg, fake_test_ind_h);

            q = sort(fake_probs_h(:, 1), 'descend');

            fake_utilities(fake_label) = ...
                merge_sort(p_if_neg, q, top_k_ind_h_if_neg, remain_budget_h);

            if fake_label == 1
                tmp_utility_if_neg = success_prob * fake_utilities(1) + ...
                    (1 - success_prob) * future_utility_if_neg_neg;
                if (pending_success_prob * future_utility_if_pos + ...
                        (1 - pending_success_prob) * tmp_utility_if_neg) <= score_l
                    tmp_pruned = true;
                    return;
                end
            end
        end
        tmp_utility_if_neg = [success_prob, 1 - success_prob] * fake_utilities;
        if (pending_success_prob * future_utility_if_pos + ...
                (1 - pending_success_prob) * tmp_utility_if_neg) <= score_l
            tmp_pruned = true;
            return;
        end

        %% if cand_ind_h is positive
        fake_utilities = zeros(2, 1);
        for fake_label = 1:2
            fake_observed_labels_l = [observed_labels_l; fake_label];
            fake_probs_h = model_h(problem, fake_train_ind_l, ...
                fake_observed_labels_l, fake_train_ind_h, ...
                fake_observed_labels_h_if_pos, fake_test_ind_h);

            q = sort(fake_probs_h(:, 1), 'descend');

            fake_utilities(fake_label) = ...
                merge_sort(p_if_pos, q, top_k_ind_h_if_pos, remain_budget_h);

            if fake_label == 1
                tmp_utility_if_pos = success_prob * fake_utilities(1) + ...
                    (1 - success_prob) * future_utility_if_pos_neg;
                if (pending_success_prob * tmp_utility_if_pos + ...
                        (1 - pending_success_prob) * tmp_utility_if_neg) <= score_l
                    tmp_pruned = true;
                    return;
                end
            end
        end
        tmp_utility_if_pos = [success_prob, 1 - success_prob] * fake_utilities;

        tmp_utility = pending_success_prob * tmp_utility_if_pos + ...
            (1 - pending_success_prob) * tmp_utility_if_neg;
    end
end

if ~exist('limit',       'var'), limit       = Inf; end
if ~exist('num_samples', 'var'), num_samples = 0;   end

num_computed = 0;
num_pruned1  = 0;
num_pruned2  = 0;

num_points        = problem.num_points;
max_num_influence = problem.max_num_influence;
remain_budget_h   = fix((problem.budget - problem.time) / problem.k);

%% unlabeled points on L
probs_l = model_l(problem, train_ind_l, observed_labels_l, unlabeled_ind_l);
probs_l = probs_l(:, 1);

num_tests_l    = numel(unlabeled_ind_l);

reverse_ind_l = zeros(num_points, 1);
reverse_ind_l(unlabeled_ind_l) = 1:num_tests_l;

%% unlabeled points on H
fake_train_ind_h              = [train_ind_h; cand_ind_h];
fake_observed_labels_h_if_neg = [observed_labels_h; 2];
fake_observed_labels_h_if_pos = [observed_labels_h; 1];

unlabeled_ind_h = unlabeled_selector(problem, fake_train_ind_h, []);
num_tests_h     = numel(unlabeled_ind_h);
reverse_ind_h   = zeros(num_points, 1);
reverse_ind_h(unlabeled_ind_h) = 1:num_tests_h;

other_weights(fake_train_ind_h, :) = 0;

%% pruning
% if cand_ind_h if negative
probs_h_if_neg = model_h(problem, train_ind_l, observed_labels_l, ...
    fake_train_ind_h, fake_observed_labels_h_if_neg, unlabeled_ind_h);
probs_h_if_neg = probs_h_if_neg(:, 1);
[~, top_k_ind_h_if_neg] = sort(probs_h_if_neg, 'descend');

prob_upper_bound_if_neg = probability_bound_h(problem, train_ind_l, ...
    observed_labels_l, fake_train_ind_h, fake_observed_labels_h_if_neg, ...
    unlabeled_ind_l, unlabeled_ind_h, 1, 0, remain_budget_h);

future_utility_if_neg_neg = ...
    sum(probs_h_if_neg(top_k_ind_h_if_neg(1:remain_budget_h)));
if max_num_influence + 1 > remain_budget_h
    future_utility_if_neg_pos = sum(prob_upper_bound_if_neg(1:remain_budget_h));
else
    tmp_ind = top_k_ind_h_if_neg(1:(remain_budget_h - max_num_influence - 1));

    future_utility_if_neg_pos = sum(probs_h_if_neg(tmp_ind)) + ...
        sum(prob_upper_bound_if_neg(1:(max_num_influence + 1)));
end

future_utility_if_neg = probs_l * future_utility_if_neg_pos + ...
    (1 - probs_l) * future_utility_if_neg_neg;

% if cand_ind_h is positive
probs_h_if_pos = model_h(problem, train_ind_l, observed_labels_l, ...
    fake_train_ind_h, fake_observed_labels_h_if_pos, unlabeled_ind_h);
probs_h_if_pos = probs_h_if_pos(:, 1);
[~, top_k_ind_h_if_pos] = sort(probs_h_if_pos, 'descend');

prob_upper_bound_if_pos = probability_bound_h(problem, train_ind_l, ...
    observed_labels_l, fake_train_ind_h, fake_observed_labels_h_if_pos, ...
    unlabeled_ind_l, unlabeled_ind_h, 1, 0, remain_budget_h);

future_utility_if_pos_neg = ...
    sum(probs_h_if_pos(top_k_ind_h_if_pos(1:remain_budget_h)));
if max_num_influence + 1 > remain_budget_h
    future_utility_if_pos_pos = sum(prob_upper_bound_if_pos(1:remain_budget_h));
else
    tmp_ind = top_k_ind_h_if_pos(1:(remain_budget_h - max_num_influence - 1));

    future_utility_if_pos_pos = sum(probs_h_if_pos(tmp_ind)) + ...
        sum(prob_upper_bound_if_pos(1:(max_num_influence + 1)));
end

future_utility_if_pos = probs_l * future_utility_if_pos_pos + ...
    (1 - probs_l) * future_utility_if_pos_neg;

pending_success_prob = model_h(problem, train_ind_l, observed_labels_l, ...
    train_ind_h, observed_labels_h, cand_ind_h);
pending_success_prob = pending_success_prob(1);

upper_bound_of_score = pending_success_prob * future_utility_if_pos + ...
    (1 - pending_success_prob) * future_utility_if_neg;
[upper_bound_of_score, sort_ind] = sort(upper_bound_of_score, 'descend');
test_ind_l = unlabeled_ind_l(sort_ind);

pruned  = false(num_tests_l, 1);
score_l = -1;

for i = 1:num_tests_l
    if pruned(i)
        num_pruned1 = num_pruned1 + 1;
        continue;
    end
    if i > limit, break; end

    this_test_ind = test_ind_l(i);
    [tmp_utility, tmp_pruned] = ...
        one_step_ens_l_full_compute_score(i, this_test_ind);

    if tmp_pruned
        num_pruned2 = num_pruned2 + 1;
        continue;
    end

    if tmp_utility > score_l
        score_l    = tmp_utility;
        cand_ind_l = this_test_ind;

        pruned(upper_bound_of_score <= score_l) = true;
    end

    num_computed = num_computed + 1;
end

num_pruned1 = num_pruned1 + sum(pruned(i:num_tests_l));

if (i < num_tests_l) && (num_samples > 0)
    candidates = (i:num_tests_l);
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

        this_test_ind = test_ind_l(i);
        [tmp_utility, tmp_pruned] = ...
            one_step_ens_l_full_compute_score(i, this_test_ind);

        if tmp_pruned
            num_pruned2 = num_pruned2 + 1;
            continue;
        end

        if tmp_utility > score_l
            score_l    = tmp_utility;
            cand_ind_l = this_test_ind;

            pruned(upper_bound_of_score <= score_l) = true;
        end

        num_computed = num_computed + 1;
    end
end

prob_l = probs_l(reverse_ind_l(cand_ind_l));

end
