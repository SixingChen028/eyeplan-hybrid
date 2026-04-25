# %% --------

source('/Users/fred/lib/utils/R/base.r')

df <- read_csvs("results/analysis/[experiment:apr25.*]/summary/evaluation.csv")


# %% --------

df %>% 
    group_by(wm_decay, beta_e_final, experiment) %>% 
    agg(n_steps_mean) %>% 
    pivot_wider(names_from=experiment, values_from=n_steps_mean)

# %% --------

df %>% 
    group_by(arg_lamda_backup, arg_hidden_size, arg_batch_size, arg_lr, arg_lamda) %>% 
    summarise(eval_reward_mean=mean(eval_reward_mean)) %>% 
    arrange(- eval_reward_mean)

# %% --------

df %>% regress(eval_reward_mean ~ arg_lamda_backup + arg_hidden_size + arg_batch_size + arg_lr + arg_lamda) %>% coeftable

# %% --------

fig_wrap("tmp", w=4, h=2,
    df %>% 
        filter(arg_lamda_backup == 0) %>% 
        ggplot(aes(factor(arg_hidden_size), eval_reward_mean, color=factor(arg_lamda))) +
        facet_grid(arg_batch_size ~ arg_lr, labeller=label_both) +
        geom_point() +
        theme() +

    df %>% 
        filter(arg_lamda_backup == 1) %>% 
        ggplot(aes(factor(arg_hidden_size), eval_reward_mean, color=factor(arg_lamda))) +
        facet_grid(arg_batch_size ~ arg_lr, labeller=label_both) +
        geom_point() +
        theme()
)




# %% --------

df %>% 
    arrange(-arg_beta_e_init) %>% 
    filter(arg_lr < 0.003) %>% 
    ggplot(aes(factor(arg_batch_size), eval_reward_mean)) +
    facet_grid(arg_lr ~ arg_lamda, labeller=label_both) +
    # geom_point() +
    points() +
    theme()

fig()    

