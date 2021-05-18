function [probs, n_l, d_l, n_h, d_h] = mf_knn_model(problem, train_ind_l, ...
    observed_labels_l, train_ind_h, observed_labels_h, test_ind_h, weights, ...
    other_weights, alpha)

n_l = nan(numel(test_ind_h), 2);
n_h = nan(numel(test_ind_h), 2);

for i = 1:2
    n_h(:, i) = alpha(i) + ...
        sum(weights(test_ind_h, train_ind_h(observed_labels_h == i)), 2);
    n_l(:, i) = ...
        sum(other_weights(test_ind_h, train_ind_l(observed_labels_l == i)), 2);
end

d_l   = sum(n_l, 2);
d_h   = sum(n_h, 2);
probs = (problem.q .* n_l + n_h) ./ (problem.q .* d_l + d_h);
