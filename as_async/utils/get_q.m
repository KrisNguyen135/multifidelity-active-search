function q_fn = get_q(train_fn, model)

q_fn = @(problem, train_ind_l, observed_labels_l, ...
    train_ind_h, observed_labels_h) ...
    train_fn(problem, train_ind_l, observed_labels_l, ...
        train_ind_h, observed_labels_h, model);
