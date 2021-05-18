function q = train_mf_model(problem, train_ind_l, observed_labels_l, ...
    train_ind_h, observed_labels_h, model)

    old_q  = problem.q;

    qs     = linspace(0.01, 1, 100);
    num_qs = numel(qs);
    loss   = nan(num_qs, 1);
    mask   = (observed_labels_h == 1);

    for q_id = 1:num_qs
      problem.q = qs(q_id);
      probs = model(problem, train_ind_l, observed_labels_l, train_ind_h, ...
                    observed_labels_h, train_ind_h);

      target_probs = probs(:, 1);
      losses(q_id) = - sum(mask .* log(target_probs) ...
                           + (1 - mask) .* log(1 - target_probs));
    end

    problem.q      = old_q;
    [~, best_q_id] = min(losses);
    q              = qs(best_q_id);
