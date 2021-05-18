% MF-ENS on L
function [cand_ind_l, prob_l, num_computed, num_pruned1, num_pruned2] = ...
    replace_shortened_l(problem, train_ind_l, observed_labels_l, ...
    train_ind_h, observed_labels_h, unlabeled_ind_l, unlabeled_ind_h, ...
    cand_ind_h, model_h, model_l, weights, other_weights, probability_bound_h, ...
    probability_bound_l, limit, num_samples, num_label_samples)

function [tmp_utility, tmp_pruned] = ...
    replace_shortened_l_compute_score(i, this_test_ind)

    tmp_utility          = -1;
    this_reverse_ind     = reverse_ind_l(this_test_ind);
    fake_unlabeled_ind_l = unlabeled_ind_l;
    fake_unlabeled_ind_l(this_reverse_ind) = [];
    success_prob         = probs_l(this_reverse_ind);

    tmp_n = n;
    tmp_d = d;
    tmp_n(this_reverse_ind, :) = [];
    tmp_d(this_reverse_ind, :) = [];

    %% if cand_ind_h is negative
    tmp_n_l = n_l_neg;
    tmp_d_l = d_l_neg;
    tmp_n_h = n_h_neg;
    tmp_d_h = d_h_neg;

    fake_utilities = zeros(2, 1);
    tmp_pruned     = false;
    for fake_label = 1:2
        [new_n_l, new_d_l, new_n_h, new_d_h, probs_h] = update_mf_knn_model( ...
            problem, tmp_n_l, tmp_d_l, tmp_n_h, tmp_d_h, this_test_ind, ...
            fake_label, true, unlabeled_ind_h, weights, other_weights);

        [new_n, new_d, new_probs_l] = update_knn_model(problem, tmp_n, tmp_d, ...
            this_test_ind, fake_label, fake_unlabeled_ind_l, weights);

        [top_probs_h, top_k_ind_h] = maxk(probs_h(:, 1), remain_budget_h);  % base ens score on H
        [min_prob_h,    min_ind_h] = min(top_probs_h);

        bottom_ind_h_mask = ~ismember((1:numel(unlabeled_ind_h))', top_k_ind_h);

        %% find top remain_budget_l on L that are not among top_k_ind_h or points observed on H
        %  should be fake_unlabeled_ind_l intersect (unlabeled_ind_h set-minus top_k_ind_h)
        mask = ismember(fake_unlabeled_ind_l, unlabeled_ind_h(bottom_ind_h_mask));

        filtered_unlabeled_ind_l   = fake_unlabeled_ind_l(mask);
        filtered_probs_l           = new_probs_l(mask, 1);

        %% additional utility by querying these top points on L
        %  only averaging over the cases where there's only one replacement

        update_mask = reverse_ind_h(filtered_unlabeled_ind_l);

        new_n_l = new_n_l(update_mask);
        new_d_l = new_d_l(update_mask);
        new_n_h = new_n_h(update_mask);
        new_d_h = new_d_h(update_mask);
        new_n_l(:, 1) = new_n_l(:, 1) + 1;
        new_d_l       = new_d_l       + 1;
        new_probs_h   = (problem.q .* new_n_l + new_n_h) ./ ...
                        (problem.q .* new_d_l + new_d_h);

        imps          = max(min_prob_h, new_probs_h(:, 1)) - min_prob_h;
        expected_imps = imps .* filtered_probs_l;
        [~, top_k_ind_l] = maxk(expected_imps, remain_budget_l);

        filtered_probs_l = filtered_probs_l(top_k_ind_l);
        new_probs_h      = new_probs_h(top_k_ind_l);

        all_probs_l = repmat(filtered_probs_l, 1, num_label_samples);
        if exist('base_label_samples', 'var')
            label_samples = base_label_samples;
        else
            label_samples = binornd(1, all_probs_l);
        end

        all_weights            = all_probs_l .* label_samples;
        fill_mask              = all_weights == 0;
        all_weights(fill_mask) = 1 - all_probs_l(fill_mask);
        sample_weights         = prod(all_weights, 1);
        sample_weights         = sample_weights / sum(sample_weights);

        utility_samples = [];
        for j = 1:num_label_samples
            new_probs_h_samples = new_probs_h .* label_samples(:, j);

            [top_probs_h, ~] = ...
                maxk([top_probs_h; new_probs_h_samples], remain_budget_h);

            utility_samples = [utility_samples; sum(top_probs_h)];
        end

        fake_utilities(fake_label) = sample_weights * utility_samples;

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
    tmp_n_l = n_l_pos;
    tmp_d_l = d_l_pos;
    tmp_n_h = n_h_pos;
    tmp_d_h = d_h_pos;

    fake_utilities = zeros(2, 1);
    for fake_label = 1:2
        [new_n_l, new_d_l, new_n_h, new_d_h, probs_h] = update_mf_knn_model( ...
            problem, tmp_n_l, tmp_d_l, tmp_n_h, tmp_d_h, this_test_ind, ...
            fake_label, true, unlabeled_ind_h, weights, other_weights);

        [new_n, new_d, new_probs_l] = update_knn_model(problem, tmp_n, tmp_d, ...
            this_test_ind, fake_label, fake_unlabeled_ind_l, weights);

        [top_probs_h, top_k_ind_h] = maxk(probs_h(:, 1), remain_budget_h);
        [min_prob_h,    min_ind_h] = min(top_probs_h);

        bottom_ind_h_mask = ~ismember((1:numel(unlabeled_ind_h))', top_k_ind_h);
        mask = ismember(fake_unlabeled_ind_l, unlabeled_ind_h(bottom_ind_h_mask));

        filtered_unlabeled_ind_l   = fake_unlabeled_ind_l(mask);
        filtered_probs_l           = new_probs_l(mask, 1);

        update_mask = reverse_ind_h(filtered_unlabeled_ind_l);

        new_n_l = new_n_l(update_mask);
        new_d_l = new_d_l(update_mask);
        new_n_h = new_n_h(update_mask);
        new_d_h = new_d_h(update_mask);
        new_n_l(:, 1) = new_n_l(:, 1) + 1;
        new_d_l       = new_d_l       + 1;
        new_probs_h   = (problem.q .* new_n_l + new_n_h) ./ ...
                        (problem.q .* new_d_l + new_d_h);

        imps          = max(min_prob_h, new_probs_h(:, 1)) - min_prob_h;
        expected_imps = imps .* filtered_probs_l;
        [~, top_k_ind_l] = maxk(expected_imps, remain_budget_l);

        filtered_probs_l = filtered_probs_l(top_k_ind_l);
        new_probs_h      = new_probs_h(top_k_ind_l);

        all_probs_l = repmat(filtered_probs_l, 1, num_label_samples);
        if exist('base_label_samples', 'var')
            label_samples = base_label_samples;
        else
            label_samples = binornd(1, all_probs_l);
        end

        all_weights            = all_probs_l .* label_samples;
        fill_mask              = all_weights == 0;
        all_weights(fill_mask) = 1 - all_probs_l(fill_mask);
        sample_weights         = prod(all_weights, 1);
        sample_weights         = sample_weights / sum(sample_weights);

        utility_samples = [];
        for j = 1:num_label_samples
            new_probs_h_samples = new_probs_h .* label_samples(:, j);

            [top_probs_h, ~] = ...
                maxk([top_probs_h; new_probs_h_samples], remain_budget_h);

            utility_samples = [utility_samples; sum(top_probs_h)];
        end

        fake_utilities(fake_label) = sample_weights * utility_samples;

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

if ~exist('limit',             'var'), limit             = Inf; end
if ~exist('num_samples',       'var'), num_samples       = 0;   end
if ~exist('num_label_samples', 'var'), num_label_samples = 32;  end

num_computed = 0;
num_pruned1  = 0;
num_pruned2  = 0;

num_points        = problem.num_points;
max_num_influence = problem.max_num_influence;
remain_budget_h   = fix((problem.budget - problem.time) / problem.k);
% remain_budget_l   = problem.budget - problem.time - problem.k;
remain_budget_l   = mod(- problem.time, problem.k);

if remain_budget_l == 0
    [cand_ind_l, prob_l, num_computed, num_pruned1, num_pruned2] = ...
        one_step_ens_l_full(problem, train_ind_l, observed_labels_l, ...
        train_ind_h, observed_labels_h, unlabeled_ind_l, unlabeled_ind_h, ...
        cand_ind_h, model_h, model_l, weights, other_weights, ...
        probability_bound_h, limit, num_samples);
    return;
end

total_num_samples = 2 ^ remain_budget_l;
if total_num_samples <= num_label_samples
    % sample matrix of size k x num_label_samples
    % each column is one sample
    base_label_samples = (dec2bin(0:(total_num_samples - 1)) - '0')';
    num_label_samples  = total_num_samples;
end

%% unlabeled points on L
[probs_l, n, d] = ...
    model_l(problem, train_ind_l, observed_labels_l, unlabeled_ind_l);
probs_l = probs_l(:, 1);
[~, top_ind_l] = sort(probs_l, 'descend');
test_ind_l     = unlabeled_ind_l(top_ind_l);

num_tests_l   = numel(unlabeled_ind_l);
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

% if cand_ind_h is negative
[probs_h_if_neg, n_l_neg, d_l_neg, n_h_neg, d_h_neg] = model_h(problem, ...
    train_ind_l, observed_labels_l, fake_train_ind_h, ...
    fake_observed_labels_h_if_neg, unlabeled_ind_h);
probs_h_if_neg = probs_h_if_neg(:, 1);
[~, top_ind_h_if_neg] = sort(probs_h_if_neg, 'descend');

% if cand_ind_h is positive
[probs_h_if_pos, n_l_pos, d_l_pos, n_h_pos, d_h_pos] = model_h(problem, ...
    train_ind_l, observed_labels_l, fake_train_ind_h, ...
    fake_observed_labels_h_if_pos, unlabeled_ind_h);
probs_h_if_pos = probs_h_if_pos(:, 1);
[~, top_ind_h_if_pos] = sort(probs_h_if_pos, 'descend');

pending_success_prob = model_h(problem, train_ind_l, observed_labels_l, ...
    train_ind_h, observed_labels_h, cand_ind_h);
pending_success_prob = pending_success_prob(1);

%% pruning
num_l_raises = min(remain_budget_h, remain_budget_l);

% if cand_ind_h is negative
prob_upper_bound_2_pos = probability_bound_h(problem, train_ind_l, ...
    observed_labels_l, fake_train_ind_h, fake_observed_labels_h_if_neg, ...
    unlabeled_ind_l, unlabeled_ind_h, remain_budget_l + 1, 0, num_l_raises);
prob_upper_bound_2_neg = probability_bound_h(problem, train_ind_l, ...
    observed_labels_l, fake_train_ind_h, fake_observed_labels_h_if_neg, ...
    unlabeled_ind_l, unlabeled_ind_h, remain_budget_l, 0, num_l_raises);

if remain_budget_h > num_l_raises
    prob_upper_bound_1 = probability_bound_h(problem, train_ind_l, ...
        observed_labels_l, fake_train_ind_h, fake_observed_labels_h_if_neg, ...
        unlabeled_ind_l, unlabeled_ind_h, 1, 0, remain_budget_h - num_l_raises);

    future_utility_if_neg_neg = ...
        sum(probs_h_if_neg(top_ind_h_if_neg(1:(remain_budget_h - num_l_raises)))) + ...
        sum(prob_upper_bound_2_neg(1:num_l_raises));

    if max_num_influence >= remain_budget_h - num_l_raises
        future_utility_if_neg_pos = ...
            sum(prob_upper_bound_1(1:(remain_budget_h - num_l_raises)));
    else
        tmp_ind = top_ind_h_if_neg(1:(remain_budget_h - num_l_raises));
        future_utility_if_neg_pos = sum(probs_h_if_neg(tmp_ind)) + ...
            sum(prob_upper_bound_1(1:max_num_influence));
    end
    future_utility_if_neg_pos = future_utility_if_neg_pos + ...
        sum(prob_upper_bound_2_pos(1:num_l_raises));
else
    future_utility_if_neg_neg = sum(prob_upper_bound_2_neg(1:num_l_raises));
    future_utility_if_neg_pos = sum(prob_upper_bound_2_pos(1:num_l_raises));
end
future_utility_if_neg = probs_l * future_utility_if_neg_pos + ...
    (1 - probs_l) * future_utility_if_neg_neg;

% if cand_ind_h is positive
prob_upper_bound_2_pos = probability_bound_h(problem, train_ind_l, ...
    observed_labels_l, fake_train_ind_h, fake_observed_labels_h_if_pos, ...
    unlabeled_ind_l, unlabeled_ind_h, remain_budget_l + 1, 0, num_l_raises);
prob_upper_bound_2_neg = probability_bound_h(problem, train_ind_l, ...
    observed_labels_l, fake_train_ind_h, fake_observed_labels_h_if_pos, ...
    unlabeled_ind_l, unlabeled_ind_h, remain_budget_l, 0, num_l_raises);

if remain_budget_h > num_l_raises
    prob_upper_bound_1 = probability_bound_h(problem, train_ind_l, ...
        observed_labels_l, fake_train_ind_h, fake_observed_labels_h_if_pos, ...
        unlabeled_ind_l, unlabeled_ind_h, 1, 0, remain_budget_h - num_l_raises);

    future_utility_if_pos_neg = ...
        sum(probs_h_if_pos(top_ind_h_if_pos(1:(remain_budget_h - num_l_raises)))) + ...
        sum(prob_upper_bound_2_neg(1:num_l_raises));

    if max_num_influence >= remain_budget_h - num_l_raises
        future_utility_if_pos_pos = ...
            sum(prob_upper_bound_1(1:(remain_budget_h - num_l_raises)));
    else
        tmp_ind = top_ind_h_if_pos(1:(remain_budget_h - num_l_raises));
        future_utility_if_pos_pos = sum(probs_h_if_pos(tmp_ind)) + ...
            sum(prob_upper_bound_1(1:max_num_influence));
    end
    future_utility_if_pos_pos = future_utility_if_pos_pos + ...
        sum(prob_upper_bound_2_pos(1:num_l_raises));
else
    future_utility_if_pos_neg = sum(prob_upper_bound_2_neg(1:num_l_raises));
    future_utility_if_pos_pos = sum(prob_upper_bound_2_pos(1:num_l_raises));
end
future_utility_if_pos = probs_l * future_utility_if_pos_pos + ...
    (1 - probs_l) * future_utility_if_pos_neg;

upper_bound_of_score = pending_success_prob * future_utility_if_pos + ...
    (1 - pending_success_prob) * future_utility_if_neg;
upper_bound_of_score = upper_bound_of_score(top_ind_l);

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
        replace_shortened_l_compute_score(i, this_test_ind);

    if tmp_pruned
        num_pruned2 = num_pruned2 + 1;
        continue;
    end

    if tmp_utility > score_l
        score_l    = tmp_utility;
        cand_ind_l = this_test_ind;

        pruned(upper_bound_of_score <= score_l) = true;
    end

    if tmp_utility > upper_bound_of_score(i)
        fprintf('=====================================\n');
        fprintf('%.5f < %.5f\n', tmp_utility, upper_bound_of_score(i));
        fprintf('=====================================\n');
    end

    num_computed = num_computed + 1;
end

num_pruned1 = num_pruned1 + sum(pruned(i:num_tests_l));

if i < num_tests_l && num_samples > 0
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
            replace_shortened_l_compute_score(i, this_test_ind);

        if tmp_pruned
            num_pruned2 = num_pruned2 + 1;
            continue;
        end

        if tmp_utility > score_l
            score_l    = tmp_utility;
            cand_ind_l = this_test_ind;

            pruned(upper_bound_of_score <= score_l) = true;
        end

        if tmp_utility > upper_bound_of_score(i)
            fprintf('=====================================\n');
            fprintf('%.5f < %.5f\n', tmp_utility, upper_bound_of_score(i));
            fprintf('=====================================\n');
        end

        num_computed = num_computed + 1;
    end
end

prob_l = probs_l(reverse_ind_l(cand_ind_l));

end
