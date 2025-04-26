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


# # normalize the values enter 0 a 1
# df_results_ft <- df_results_ft |> 
#   dplyr::mutate(
#     dplyr::across(
#       c(loss_train, loss_eval),
#       ~ scales::rescale(., to = c(0, 1), from = range(., na.rm = TRUE))
#     )
#   )

ggplot(df_results_ft, aes(x = step)) +
  geom_line(aes(y = loss_train, color = "Train"),
            size = 1.2, lineend = "round") +            # thicker lines, rounded ends :contentReference[oaicite:3]{index=3}
  geom_line(aes(y = loss_eval,  color = "Eval"),
            size = 1.2, linetype = "dashed") +       # dashed for eval loss :contentReference[oaicite:4]{index=4}
  
  # 3. Facet by model
  facet_wrap(~ model, scales = "free_y", ncol = 1) +  # one column of panels, independent y-scales :contentReference[oaicite:5]{index=5}
  
  # 4. Color scales
  scale_color_manual(
    values = c("Train" = "#1b9e77", "Eval" = "#d95f02"),
    name   = "Loss Type"
  ) +                                                 # manual color mapping :contentReference[oaicite:6]{index=6}
  
  # 5. Minimal theme with custom tweaks
  theme_minimal(base_size = 14) +                     # clean, minimal background :contentReference[oaicite:7]{index=7}
  theme(
    strip.text       = element_text(face = "bold", size = 16),
    axis.title       = element_text(face = "bold", size = 14),
    legend.position  = "top",
    legend.title     = element_blank(),
    panel.grid.major = element_line(linetype = "dotted", colour = "grey80"),
    panel.grid.minor = element_blank()
  ) +
  
  # 6. Labels
  labs(
    title    = "Training vs. Evaluation Loss per Model",
    subtitle = "Faceted line plots of loss over training steps",
    x        = "Training Step",
    y        = "Loss"
  )

