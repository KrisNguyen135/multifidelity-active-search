% MF-ENS on H
function [cand_ind_h, prob_h, num_computed, num_pruned1, num_pruned2] = ...
    replace_shortened_h(problem, train_ind_l, observed_labels_l, ...
    train_ind_h, observed_labels_h, unlabeled_ind_l, unlabeled_ind_h, ...
    model_h, model_l, weights, other_weights, probability_bound_h, ...
    probability_bound_l, limit, num_samples, num_label_samples)

function [tmp_utility, tmp_pruned] = ...
    replace_shortened_h_compute_score(i, this_test_ind)

    tmp_utility          = -1;
    this_reverse_ind     = reverse_ind_h(this_test_ind);
    fake_unlabeled_ind_h = unlabeled_ind_h;
    fake_unlabeled_ind_h(this_reverse_ind) = [];
    tmp_reverse_ind_h    = zeros(num_points, 1);
    tmp_reverse_ind_h(fake_unlabeled_ind_h) = 1:(num_tests_h - 1);
    success_prob         = probs_h(this_reverse_ind);

    tmp_n_l = n_l;
    tmp_d_l = d_l;
    tmp_n_h = n_h;
    tmp_d_h = d_h;
    tmp_n_l(this_reverse_ind, :) = [];
    tmp_d_l(this_reverse_ind, :) = [];
    tmp_n_h(this_reverse_ind, :) = [];
    tmp_d_h(this_reverse_ind, :) = [];

    fake_utilities = zeros(2, 1);
    tmp_pruned     = false;
    for fake_label = 1:2
        [new_n_l, new_d_l, new_n_h, new_d_h, new_probs_h] = update_mf_knn_model( ...
            problem, tmp_n_l, tmp_d_l, tmp_n_h, tmp_d_h, this_test_ind, ...
            fake_label, false, fake_unlabeled_ind_h, weights, other_weights);

        [top_probs_h, top_k_ind_h] = maxk(new_probs_h(:, 1), remain_budget_h);  % base ens score on H
        [min_prob_h,    min_ind_h] = min(top_probs_h);

        bottom_ind_h_mask = ~ismember((1:numel(fake_unlabeled_ind_h))', top_k_ind_h);

        mask = ismember(unlabeled_ind_l, fake_unlabeled_ind_h(bottom_ind_h_mask));

        filtered_unlabeled_ind_l   = unlabeled_ind_l(mask);
        filtered_probs_l           = probs_l(mask, 1);

        update_mask = tmp_reverse_ind_h(filtered_unlabeled_ind_l);

        new_n_l = new_n_l(update_mask, :);
        new_d_l = new_d_l(update_mask);
        new_n_h = new_n_h(update_mask, :);
        new_d_h = new_d_h(update_mask);
        new_n_l(:, 1) = new_n_l(:, 1) + 1;
        new_d_l       = new_d_l       + 1;
        new_probs_h   = (problem.q .* new_n_l + new_n_h) ./ ...
                        (problem.q .* new_d_l + new_d_h);

        imps          = max(min_prob_h, new_probs_h(:, 1)) - min_prob_h;
        expected_imps = imps .* filtered_probs_l;
        [~, top_k_ind_l] = maxk(expected_imps, problem.k);

        filtered_probs_l = filtered_probs_l(top_k_ind_l);  % size k x 1
        new_probs_h      = new_probs_h(top_k_ind_l);       % size k x 1

        %% construct the sample and weight matrices
        all_probs_l = repmat(filtered_probs_l, 1, num_label_samples);   % size k x num_label_samples
        if exist('base_label_samples', 'var')
            label_samples = base_label_samples;                         % size k x num_label_samples
        else  % draw independenct samples
            label_samples = binornd(1, all_probs_l);                    % size k x num_label_samples
        end

        all_weights            = all_probs_l .* label_samples;          % size k x num_label_samples
        fill_mask              = all_weights == 0;
        all_weights(fill_mask) = 1 - all_probs_l(fill_mask);
        sample_weights         = prod(all_weights, 1);                  % size 1 x num_label_samples
        sample_weights         = sample_weights / sum(sample_weights);  % size 1 x num_label_samples

        %% merge sort the two probability arrays for each sample
        utility_samples = [];
        for j = 1:num_label_samples
            new_probs_h_samples = new_probs_h .* label_samples(:, j);

            [top_probs_h, ~] = ...
                maxk([top_probs_h; new_probs_h_samples], remain_budget_h);

            utility_samples = [utility_samples; sum(top_probs_h)];
        end

        fake_utilities(fake_label) = sample_weights * utility_samples;

        if fake_label == 1 && (success_prob * (fake_utilities(1) + 1) + ...
                (1 - success_prob) * future_utility_if_neg) <= score_h
            tmp_pruned = true;
            return;
        end
    end
    tmp_utility = success_prob + ...
        [success_prob, 1 - success_prob] * fake_utilities;
end

if ~exist('limit',             'var'), limit             = Inf; end
if ~exist('num_samples',       'var'), num_samples       = 0;   end
if ~exist('num_label_samples', 'var'), num_label_samples = 32;  end

num_computed = 0;
num_pruned1  = 0;
num_pruned2  = 0;

total_num_samples = 2 ^ problem.k;
if total_num_samples <= num_label_samples
    % sample matrix of size k x num_label_samples
    % each column is one sample
    base_label_samples = (dec2bin(0:(total_num_samples - 1)) - '0')';
    num_label_samples  = total_num_samples;
end

num_points        = problem.num_points;
max_num_influence = problem.max_num_influence;
remain_budget_h   = fix((problem.budget - problem.time) / problem.k);
remain_budget_l   = problem.budget - problem.time - problem.k;

%% unlabeled points on L
[probs_l, n, d] = ...
    model_l(problem, train_ind_l, observed_labels_l, unlabeled_ind_l);
probs_l = probs_l(:, 1);

num_tests_l   = numel(unlabeled_ind_l);
reverse_ind_l = zeros(num_points, 1);
reverse_ind_l(unlabeled_ind_l) = 1:num_tests_l;

%% unlabeled points on H
num_tests_h     = numel(unlabeled_ind_h);
reverse_ind_h   = zeros(num_points, 1);
reverse_ind_h(unlabeled_ind_h) = 1:num_tests_h;

[probs_h, n_l, d_l, n_h, d_h] = model_h(problem, train_ind_l, ...
    observed_labels_l, train_ind_h, observed_labels_h, unlabeled_ind_h);
probs_h = probs_h(:, 1);
[~, top_ind_h] = sort(probs_h, 'descend');
test_ind_h     = unlabeled_ind_h(top_ind_h);

if problem.time > problem.budget - problem.k
    cand_ind_h   = test_ind_h(1);
    prob_h       = probs_h(top_ind_h(1));
    num_computed = numel(unlabeled_ind_h);
    return;
end

%% pruning
num_l_raises = min(remain_budget_h, problem.k);

prob_upper_bound_2_pos = probability_bound_h(problem, train_ind_l, ...
    observed_labels_l, train_ind_h, observed_labels_h, unlabeled_ind_l, ...
    unlabeled_ind_h, problem.k, 1, num_l_raises);
prob_upper_bound_2_neg = probability_bound_h(problem, train_ind_l, ...
    observed_labels_l, train_ind_h, observed_labels_h, unlabeled_ind_l, ...
    unlabeled_ind_h, problem.k, 0, num_l_raises);

if remain_budget_h > num_l_raises
    prob_upper_bound_1 = probability_bound_h(problem, train_ind_l, ...
        observed_labels_l, train_ind_h, observed_labels_h, unlabeled_ind_l, ...
        unlabeled_ind_h, 0, 1, remain_budget_h - num_l_raises);

    future_utility_if_neg = ...
        sum(probs_h(top_ind_h(1:(remain_budget_h - num_l_raises)))) + ...
        sum(prob_upper_bound_2_neg(1:num_l_raises));

    if max_num_influence >= remain_budget_h - num_l_raises
        future_utility_if_pos = ...
            sum(prob_upper_bound_1(1:(remain_budget_h - num_l_raises)));
    else
        tmp_ind = top_ind_h(1:(remain_budget_h - num_l_raises - max_num_influence));
        future_utility_if_pos = sum(probs_h(tmp_ind)) + ...
                                sum(prob_upper_bound_1(1:max_num_influence));
    end
    future_utility_if_pos = future_utility_if_pos + ...
                            sum(prob_upper_bound_2_pos(1:num_l_raises));
else
    future_utility_if_neg = sum(prob_upper_bound_2_neg(1:num_l_raises));
    future_utility_if_pos = sum(prob_upper_bound_2_pos(1:num_l_raises));
end

future_utility = probs_h * future_utility_if_pos + ...
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

    this_test_ind = test_ind_h(i);
    [tmp_utility, tmp_pruned] = ...
        replace_shortened_h_compute_score(i, this_test_ind);

    if tmp_pruned
        num_pruned2 = num_pruned2 + 1;
        continue;
    end

    if tmp_utility > score_h
        score_h    = tmp_utility;
        cand_ind_h = this_test_ind;

        pruned(upper_bound_of_score <= score_h) = true;
    end

    if tmp_utility > upper_bound_of_score(i)
        fprintf('=====================================\n');
        fprintf('%.5f < %.5f\n', tmp_utility, upper_bound_of_score(i));
        fprintf('=====================================\n');
    end

    num_computed = num_computed + 1;
end

num_pruned1 = num_pruned1 + sum(pruned(i:num_tests_h));

if i < num_tests_h && num_samples > 0
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

        this_test_ind = test_ind_h(i);
        [tmp_utility, tmp_pruned] = ...
            replace_shortened_h_compute_score(i, this_test_ind);

        if tmp_pruned
            num_pruned2 = num_pruned2 + 1;
            continue;
        end

        if tmp_utility > score_h
            score_h    = tmp_utility;
            cand_ind_h = this_test_ind;

            pruned(upper_bound_of_score <= score_h) = true;
        end

        if tmp_utility > upper_bound_of_score(i)
            fprintf('=====================================\n');
            fprintf('%.5f < %.5f\n', tmp_utility, upper_bound_of_score(i));
            fprintf('=====================================\n');
        end

        num_computed = num_computed + 1;
    end
end

prob_h = probs_h(reverse_ind_h(cand_ind_h));

end
