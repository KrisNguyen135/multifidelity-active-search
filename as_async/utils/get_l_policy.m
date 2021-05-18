function l_policy = get_l_policy(policy, varargin)

l_policy = @(problem, train_ind_l, observed_labels_l, train_ind_h, ...
    observed_labels_h, unlabeled_ind_l, unlabeled_ind_h, cand_ind_h) ...
    policy(problem, train_ind_l, observed_labels_l, train_ind_h, ...
        observed_labels_h, unlabeled_ind_l, unlabeled_ind_h, cand_ind_h, ...
        varargin{:});