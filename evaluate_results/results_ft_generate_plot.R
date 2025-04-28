library(ggplot2)

df_results_ft <- readr::read_csv("experiments/experiments_wandb_metadata/losses_per_step_all_models.csv")

df_results_ft <- df_results_ft |> 
  dplyr::mutate(
    model_name = dplyr::case_when(
    stringr::str_detect(model, '(?i)Phi-4') ~ 'Phi-4',
    stringr::str_detect(model, '(?i)Llama-3.2-3B') ~ 'Llama-3.2-3B',
    stringr::str_detect(model, '(?i)sabia-7b') ~ 'Sabiá-7B',
    TRUE ~ model
    ),
  ) |> 
  dplyr::relocate(model_name) |> 
  dplyr::select(-model) |> 
  dplyr::rename(model = model_name)

# remove sabiá
df_results_ft <- df_results_ft |> 
  dplyr::filter(model != 'Sabiá-7B')

# normalize the scales to 0 - 1

df_plot <- df_results_ft  |> 
  dplyr::arrange(model, step) |>                   # make sure steps are in increasing order
  dplyr::group_by(model) |>
  # 1) carry last non-NA LOSS_EVAL forward
  tidyr::fill(loss_eval, .direction = "down") |>
  # 2) carry the first non-NA backward (for any leading NAs)
  tidyr::fill(loss_eval, .direction = "up") |>
  # 3) min–max normalize PER MODEL
  dplyr::mutate(
    train_norm = (loss_train - min(loss_train, na.rm=TRUE)) /
      (max(loss_train, na.rm=TRUE) - min(loss_train, na.rm=TRUE)),
    eval_norm  = (loss_eval  - min(loss_eval,  na.rm=TRUE)) /
      (max(loss_eval,  na.rm=TRUE) - min(loss_eval,  na.rm=TRUE))
  ) |> 
  dplyr::ungroup()

# 4) plot train & eval on the same axes, one facet per model
plot <- ggplot(df_plot, aes(x = step)) +
  geom_line(aes(y = train_norm, color = "Train Loss"), linewidth = 0.8) +
  geom_line(aes(y = eval_norm,  color = "Eval  Loss"), linewidth = 0.8) +
  facet_wrap(~model, scales = "fixed", ncol = 1) +
  scale_color_manual(
    name   = NULL,
    values = c("Train Loss" = "steelblue", "Eval  Loss" = "firebrick")
  ) +
  scale_x_continuous(
    breaks = seq(0, max(df_plot$step), by = 50)
  ) +
  labs(
    title = "Training and Evaluation Losses in the Fine-Tunning Process",
    x     = "Training Step",
    y     = "Normalized Loss (0–1)"
  ) +
    #caption = "Losses are normalized per model to 0–1. The values were carried forward and backward to fill the gaps in the training steps.") +
  theme_minimal(base_size = 13) +
  theme(legend.position = "bottom")

ggsave(plot = plot, filename = 'evaluate_results/finetune_losses.png', dpi = 900, width = 12, height = 16, units = "in")
