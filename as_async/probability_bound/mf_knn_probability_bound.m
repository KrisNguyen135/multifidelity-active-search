function bounds = mf_knn_probability_bound(problem, ...
    train_ind_l, observed_labels_l, train_ind_h, observed_labels_h, ...
    test_ind_l, test_ind_h, num_positives_l, num_positives_h, ...
    remain_budget_h, weights, other_weights, knn_ind, other_knn_ind, ...
    knn_weights, other_knn_weights, alpha, sort_probs)

if ~exist('sort_probs', 'var'), sort_probs = true; end

num_tests = numel(test_ind_h);

positive_ind_l = (observed_labels_l == 1);
successes_l    = sum(other_weights(test_ind_h, train_ind_l( positive_ind_l)), 2);
failures_l     = sum(other_weights(test_ind_h, train_ind_l(~positive_ind_l)), 2);

positive_ind_h = (observed_labels_h == 1);
successes_h    = sum(weights(test_ind_h, train_ind_h( positive_ind_h)), 2);
failures_h     = sum(weights(test_ind_h, train_ind_h(~positive_ind_h)), 2);

in_other_train = ismember(other_knn_ind(:), train_ind_l);
other_knn_weights(in_other_train) = 0;
in_train = ismember(knn_ind(:), train_ind_h);
knn_weights(in_train) = 0;

if num_positives_l == 0
    success_count_bound_l = 0;
elseif num_positives_l == 1
    success_count_bound_l = max(other_knn_weights(test_ind_h, ...
        1:min(end, length(train_ind_l) + num_positives_l)), [], 2);
else
    success_count_bound_l = nan(num_tests, 1);
    for i = 1:num_tests
        tmp_test_ind = test_ind_h(i);
        row_positive_ind = find(other_knn_weights(tmp_test_ind, :), num_positives_l);
        success_count_bound_l(i) = sum(other_knn_weights(tmp_test_ind, row_positive_ind));
    end
end

if num_positives_h == 0
    success_count_bound_h = 0;
elseif num_positives_h == 1
    success_count_bound_h = max(knn_weights(test_ind_h, ...
        1:min(end, length(train_ind_h) + num_positives_h)), [], 2);
else
    success_count_bound_h = nan(num_tests, 1);
    for i = 1:num_tests
        tmp_test_ind = test_ind_h(i);
        row_positive_ind = find(knn_weights(tmp_test_ind, :), num_positives_h);
        success_count_bound_h(i) = sum(knn_weights(tmp_test_ind, row_positive_ind));
    end
end

max_alpha = alpha(1) ...
            +  successes_h + success_count_bound_h ...
            + (successes_l + success_count_bound_l) * problem.q;
min_beta  = alpha(2) + failures_h + failures_l * problem.q;

probs = max_alpha ./ (max_alpha + min_beta);
if remain_budget_h <= 1
    bounds = max(probs);
else
    if sort_probs, probs = sort(probs, 'descend'); end
    bounds = probs(1:remain_budget_h);
end
