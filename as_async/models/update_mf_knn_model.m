function [n_l, d_l, n_h, d_h, p] = update_mf_knn_model(problem, n_l, d_l, ...
    n_h, d_h, new_train_ind, new_label, on_l, test_ind, weights, other_weights)

% fprintf('%d %d %d %d\n', size(n_h, 1), size(d_h, 1), size(n_l, 1), ...
%     size(d_l, 1));

if on_l
    new_vector        = other_weights(test_ind, new_train_ind);
    n_l(:, new_label) = n_l(:, new_label) + new_vector;
    d_l               = d_l               + new_vector;
else
    new_vector        = weights(test_ind, new_train_ind);
    n_h(:, new_label) = n_h(:, new_label) + new_vector;
    d_h               = d_h               + new_vector;
end

p = (problem.q .* n_l + n_h) ./ (problem.q .* d_l + d_h);

% fprintf('%d %d %d %d %d\n', size(n_h, 1), size(d_h, 1), size(n_l, 1), ...
%     size(d_l, 1), size(new_vector, 1));
