# ============================================================
# 0. Libraries
# ============================================================
library(tidyverse)
library(readxl)
library(janitor)
library(stringr)
library(purrr)

# ============================================================
# 1. Utility: numeric cleaner
# ============================================================
clean_num <- function(x) {
  x <- as.character(x)
  x <- str_replace_all(x, "[^0-9\\.\\-]", "")
  suppressWarnings(as.numeric(x))
}

# ============================================================
# 2. Locate all ALL_MODELS files
# ============================================================
dir_tf  <- "/Users/taircarmon/Desktop/Uni-Projects/prawn-size-project/scripts/eval/outputs_TF"
dir_out <- "/Users/taircarmon/Desktop/Uni-Projects/prawn-size-project/scripts/eval/outputs"

files_tf  <- list.files(dir_tf,  pattern = "all_models.*\\.xlsx$", recursive = TRUE, full.names = TRUE)
files_out <- list.files(dir_out, pattern = "all_models.*\\.xlsx$", recursive = TRUE, full.names = TRUE)

files <- c(files_tf, files_out)

# ============================================================
# 3. Helper: infer test year from file path
# ============================================================
infer_year_from_path <- function(path) {
  if (str_detect(path, "test_on_2024")) return(2024L)
  if (str_detect(path, "test_on_2025")) return(2025L)
  NA_integer_
}

# ============================================================
# 4. Read & standardize a single ALL_MODELS file
# ============================================================
process_file <- function(path) {
  
  year <- infer_year_from_path(path)
  df <- read_excel(path, sheet = "ALL_MODELS") %>% clean_names()
  source_folder <- ifelse(str_detect(path, "outputs_TF"), "outputs_tf", "outputs")
  
  # ---------------------------
  # Case A: 2024 — wide layout
  # ---------------------------
  if (year == 2024) {
    
    stopifnot(all(c("model", "metric") %in% names(df)))
    
    df_2024 <- df %>%
      mutate(
        metric = str_squish(as.character(metric)),
        model  = str_squish(as.character(model)),
        combined = clean_num(combined)
      ) %>%
      mutate(
        metric_std = case_when(
          str_detect(metric, "MAE carapace") ~ "MAE_CL",
          str_detect(metric, "MPE carapace") ~ "MPE_CL",
          str_detect(metric, "MAE total")    ~ "MAE_TL",
          str_detect(metric, "MPE total")    ~ "MPE_TL",
          str_detect(metric, "Detection Rate") & str_detect(metric, "carapace") ~ "DR_CL",
          str_detect(metric, "Detection Rate") & str_detect(metric, "total")    ~ "DR_TL",
          TRUE ~ NA_character_
        )
      ) %>%
      filter(!is.na(metric_std)) %>%
      select(model, metric_std, combined) %>%
      pivot_wider(names_from = metric_std, values_from = combined) %>%
      mutate(
        year = year,
        test_year = year,
        file = basename(path),
        source = source_folder
      )
    
    return(df_2024)
  }
  
  # ---------------------------
  # Case B: 2025 — flat layout
  # ---------------------------
  if (year == 2025) {
    
    stopifnot("model" %in% names(df))
    
    col_dr_cl  <- names(df)[str_detect(names(df), "detection_rate.*carapace")]
    col_dr_tl  <- names(df)[str_detect(names(df), "detection_rate.*total")]
    col_mae_cl <- names(df)[str_detect(names(df), "mae.*carapace")]
    col_mpe_cl <- names(df)[str_detect(names(df), "mpe.*carapace")]
    col_mae_tl <- names(df)[str_detect(names(df), "mae.*total")]
    col_mpe_tl <- names(df)[str_detect(names(df), "mpe.*total")]
    
    df_2025 <- df %>%
      mutate(model = str_squish(as.character(model))) %>%
      transmute(
        model,
        DR_CL  = if (length(col_dr_cl) > 0)  clean_num(.data[[col_dr_cl[1]]])  else NA_real_,
        DR_TL  = if (length(col_dr_tl) > 0)  clean_num(.data[[col_dr_tl[1]]])  else NA_real_,
        MAE_CL = if (length(col_mae_cl) > 0) clean_num(.data[[col_mae_cl[1]]]) else NA_real_,
        MPE_CL = if (length(col_mpe_cl) > 0) clean_num(.data[[col_mpe_cl[1]]]) else NA_real_,
        MAE_TL = if (length(col_mae_tl) > 0) clean_num(.data[[col_mae_tl[1]]]) else NA_real_,
        MPE_TL = if (length(col_mpe_tl) > 0) clean_num(.data[[col_mpe_tl[1]]]) else NA_real_,
        year = year,
        test_year = year,
        file = basename(path),
        source = source_folder
      )
    
    return(df_2025)
  }
  
  tibble()
}

# ============================================================
# 5. Run processing on all files
# ============================================================
all_results <- map_df(files, process_file)

results_2024 <- all_results %>% filter(test_year == 2024)
results_2025 <- all_results %>% filter(test_year == 2025)

print(paste("Rows 2024:", nrow(results_2024)))
print(paste("Rows 2025:", nrow(results_2025)))

# ============================================================
# 6. Classification: precise & coarse grouping
# ============================================================

classify_precise <- function(m) {
  case_when(
    str_detect(m, "2024all_to_2025all")  ~ "CS-24→25 (All→All)",
    str_detect(m, "2024circ_to_2025all") ~ "CS-24→25 (Circ→All)",
    str_detect(m, "2025all_to_2024all")  ~ "CS-25→24 (All→All)",
    str_detect(m, "2025all_to_2024circ") ~ "CS-25→24 (All→Circ)",
    
    str_detect(m, "circular_to_square")  ~ "WS_24_Circ_to_Square",
    str_detect(m, "square_to_circular")  ~ "WS_24_Square_to_Circ",
    
    str_detect(m, "^circulars")          ~ "Baseline-2024-Circ",
    str_detect(m, "^square")             ~ "Baseline-2024-Square",
    str_detect(m, "^all-ponds")          ~ "Baseline-2024-All",
    str_detect(m, "2025_all") & !str_detect(m, "to") ~ "Baseline-2025-All",
    
    str_detect(m, "2024_2025")           ~ "Joint-Train",
    
    TRUE ~ NA_character_
  )
}

classify_coarse <- function(m) {
  case_when(
    str_detect(m, "b1") ~ "B1",
    str_detect(m, "b2") ~ "B2",
    str_detect(m, "b3") ~ "B3",
    str_detect(m, "2024to2025") ~ "CS-24→25 (All→All)",
    str_detect(m, "2025to2024") ~ "CS-25→24 (All→All)",
    TRUE ~ "Unknown"
  )
}

classified_df <- all_results %>%
  mutate(
    model_str = tolower(model),
    precise_group = classify_precise(model_str),
    coarse_group  = classify_coarse(model_str),
    experiment_group = ifelse(is.na(precise_group), coarse_group, precise_group),
    
    k_shot = str_extract(model_str, "k[0-9]+pct") %>% str_remove_all("k|pct") %>% as.numeric(),
    freeze = str_extract(model_str, "freeze[0-9]+") %>% str_remove("freeze") %>% as.numeric()
  )

count(classified_df, experiment_group)

# ============================================================
# 7. Convert to long format (for plotting)
# ============================================================
df_long <- classified_df %>%
  pivot_longer(
    cols = c(DR_CL, DR_TL, MAE_CL, MPE_CL, MAE_TL, MPE_TL),
    names_to = "metric",
    values_to = "value"
  ) %>%
  filter(!is.na(value))

# ============================================================
#graphs!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ============================================================
clean_names <- tribble(
  ~raw, ~clean,
  "circulars_weights_best.pt",    "Baseline 2024 – Circular",
  "square_weights_best.pt",       "Baseline 2024 – Square",
  "all-ponds_weights_best.pt",    "Baseline 2024 – All-Ponds",
  "yolov11n_train_on_2025_all_pose_300ep_best.pt", "Baseline 2025 – All",
  "yolov11n_train_on_2024_2025_all_pose_300ep_best.pt", "Joint (2024+2025)"
)

df_filtered <- classified_df %>%
  mutate(model_lower = tolower(model)) %>%
  filter(model_lower %in% clean_names$raw) %>%
  mutate(model_clean = clean_names$clean[match(model_lower, clean_names$raw)])

df_long <- df_filtered %>%
  pivot_longer(
    cols = c(MPE_CL, MPE_TL),
    names_to = "metric",
    values_to = "value"
  ) %>%
  filter(!is.na(value))

df_g1 <- df_long %>% filter(test_year == 2025)

g1_mpe <- df_g1 %>%
  ggplot(aes(x = reorder(model_clean, value), y = value, fill = metric)) +
  geom_col(position = position_dodge()) +
  coord_flip() +
  labs(
    title = "Group 1 — Baselines — MPE (Test 2025)",
    x = "Model",
    y = "MPE (%)"
  ) +
  theme_bw()

g1_mpe

# ============================================================
# GROUP 1 — Baselines — Test 2024
# ============================================================

# 1. Filter baseline best models only
df_g1_2024 <- df_filtered %>%
  pivot_longer(
    cols = c(MPE_CL, MPE_TL),
    names_to = "metric",
    values_to = "value"
  ) %>%
  filter(!is.na(value)) %>%
  mutate(test_year = 2024)  

# 2. Plot — MPE (CL + TL)
g1_2024_mpe <- df_g1_2024 %>%
  ggplot(aes(x = reorder(model_clean, value), y = value, fill = metric)) +
  geom_col(position = "dodge") +
  coord_flip() +
  labs(
    title = "Group 1 — Baselines — MPE (Test 2024)",
    x = "Model",
    y = "MPE (%)"
  ) +
  theme_bw()

g1_2024_mpe

df_g1_2025 <- df_filtered %>%
  pivot_longer(
    cols = c(MPE_CL, MPE_TL),
    names_to = "metric",
    values_to = "value"
  ) %>%
  filter(!is.na(value)) %>%
  mutate(test_year = 2025)  

# ============================================================
# GROUP 2 — Cross-season — Test 2024
# ============================================================
library(tidyverse)

# ============================================================
#  GROUP 2A — WITHIN-SEASON (WS) — extract automatically
# ============================================================

df_ws <- classified_df %>%
  filter(experiment_group %in% c("WS_24_Circ_to_Square",
                                 "WS_24_Square_to_Circ")) %>%
  mutate(model_clean = case_when(
    experiment_group == "WS_24_Circ_to_Square" ~ "WS 24 — Circular→Square",
    experiment_group == "WS_24_Square_to_Circ" ~ "WS 24 — Square→Circular",
    TRUE ~ experiment_group
  ))

# Pivot to long
df_ws_long <- df_ws %>%
  pivot_longer(
    cols = c(DR_CL, DR_TL, MAE_CL, MPE_CL, MAE_TL, MPE_TL),
    names_to = "metric",
    values_to = "value"
  ) %>%
  filter(!is.na(value))

# ----------------------------
# WS — Test on 2024
# ----------------------------
ws_2024_plot <- df_ws_long %>%
  filter(test_year == 2024) %>%
  ggplot(aes(x = reorder(model_clean, value), y = value, fill = metric)) +
  geom_col(position = "dodge") +
  coord_flip() +
  labs(
    title = "Group 2A — Within-Season (WS) — Test 2024",
    x = "Model",
    y = "Value"
  ) +
  theme_bw()

ws_2024_plot

# ----------------------------
# WS — Test on 2025
# ----------------------------
ws_2025_plot <- df_ws_long %>%
  filter(test_year == 2025) %>%
  ggplot(aes(x = reorder(model_clean, value), y = value, fill = metric)) +
  geom_col(position = "dodge") +
  coord_flip() +
  labs(
    title = "Group 2A — Within-Season (WS) — Test 2025",
    x = "Model",
    y = "Value"
  ) +
  theme_bw()

ws_2025_plot



# ============================================================
#  GROUP 2B — CROSS-SEASON (CS) — extract automatically
# ============================================================

df_cs <- classified_df %>%
  filter(str_detect(experiment_group, "CS")) %>%
  mutate(model_clean = case_when(
    experiment_group == "CS-24→25 (All→All)" ~ "CS 24→25 — All→All",
    experiment_group == "CS-24→25 (Circ→All)" ~ "CS 24→25 — Circ→All",
    experiment_group == "CS-25→24 (All→All)" ~ "CS 25→24 — All→All",
    experiment_group == "CS-25→24 (All→Circ)" ~ "CS 25→24 — All→Circ",
    TRUE ~ experiment_group
  ))

# Pivot long
df_cs_long <- df_cs %>%
  pivot_longer(
    cols = c(DR_CL, DR_TL, MAE_CL, MPE_CL, MAE_TL, MPE_TL),
    names_to = "metric",
    values_to = "value"
  ) %>%
  filter(!is.na(value))


# ----------------------------
# CS — Test on 2024
# ----------------------------
cs_2024_plot <- df_cs_long %>%
  filter(test_year == 2024) %>%
  ggplot(aes(x = reorder(model_clean, value), y = value, fill = metric)) +
  geom_col(position = "dodge") +
  coord_flip() +
  labs(
    title = "Group 2B — Cross-Season (CS) — Test 2024",
    x = "Model",
    y = "Value"
  ) +
  theme_bw()

cs_2024_plot


# ----------------------------
# CS — Test on 2025
# ----------------------------
cs_2025_plot <- df_cs_long %>%
  filter(test_year == 2025) %>%
  ggplot(aes(x = reorder(model_clean, value), y = value, fill = metric)) +
  geom_col(position = "dodge") +
  coord_flip() +
  labs(
    title = "Group 2B — Cross-Season (CS) — Test 2025",
    x = "Model",
    y = "Value"
  ) +
  theme_bw()

cs_2025_plot
library(tidyverse)

# ============================================================
# 1. Define baselines for each test set
# ============================================================
baseline_map <- tibble(
  test_year = c(2024, 2025),
  baseline_name = c("Baseline-2024-All", "Baseline-2025-All")
)

# ============================================================
# 2. Filter only B1 / B2 / B3 models
# ============================================================
g3 <- classified_df %>%
  filter(experiment_group %in% c("B1","B2","B3")) %>%
  mutate(
    model_config = paste0(
      experiment_group,
      " | k=", k_shot,
      " | freeze=", freeze
    )
  )

# ============================================================
# 3. Pivot to long (use only MAE_CL & MAE_TL)
# ============================================================
g3_long <- g3 %>%
  pivot_longer(
    cols = c(MAE_CL, MAE_TL),
    names_to = "metric",
    values_to = "value"
  ) %>%
  filter(!is.na(value))

# ============================================================
# 4. Choose best model per group (B1/B2/B3) & test_year
#     Criterion = minimal MAE_CL
# ============================================================
best_per_group <- g3_long %>%
  filter(metric == "MAE_CL") %>%
  group_by(test_year, experiment_group) %>%
  slice_min(value, n = 1) %>%
  ungroup() %>%
  select(test_year, experiment_group, model, model_config)

# ============================================================
# 5. Get baseline values for comparison
# ============================================================
baseline_values <- classified_df %>%
  filter(model %in% baseline_map$baseline_name) %>%
  pivot_longer(
    cols = c(MAE_CL, MAE_TL),
    names_to = "metric",
    values_to = "value"
  ) %>%
  select(model, test_year, metric, value)


# ============================================================
# 6. Plotting function for each test_year
# ============================================================
plot_g3 <- function(year_selected) {
  
  baseline_name <- baseline_map$baseline_name[baseline_map$test_year == year_selected]
  
  df_plot <- g3_long %>%
    filter(test_year == year_selected)
  
  df_best <- best_per_group %>%
    filter(test_year == year_selected)
  
  # Join best values (MAE_CL only)
  df_best_values <- df_best %>%
    left_join(
      df_plot %>% filter(metric == "MAE_CL") %>% select(model_config, metric, value),
      by = "model_config"
    )
  
  df_baseline <- baseline_values %>%
    filter(test_year == year_selected)
  
  p <- df_plot %>%
    ggplot(aes(x = reorder(model_config, value), y = value, fill = metric)) +
    geom_col(position = "dodge") +
    coord_flip() +
    
    # ---- best model markers ----
  geom_point(
    data = df_best_values,
    aes(x = model_config, y = value),
    color = "red", size = 5, shape = 8
  ) +
    
    # ---- baseline dashed line ----
  geom_hline(
    data = df_baseline,
    aes(yintercept = value, color = metric),
    linetype = "dashed", linewidth = 1
  ) +
    
    labs(
      title = paste0("Group 3 — B1/B2/B3 — Test ", year_selected),
      x = "Model (k & freeze configuration)",
      y = "MAE (mm)",
      subtitle = paste("Baseline:", baseline_name)
    ) +
    theme_bw() +
    theme(text = element_text(size = 12))
  
  return(p)
}


# ============================================================
# 7. Produce both plots
# ============================================================
g3_plot_2024 <- plot_g3(2024)
g3_plot_2025 <- plot_g3(2025)

# Show plots
g3_plot_2024
g3_plot_2025

# ============================================================
# new day new graphs!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ============================================================
# ============================================================
# NEW PLOTTING SECTION – FOR PAPER FIGURES
# ============================================================
library(tidyverse)

# ---------- Theme & colors (Elsevier-ish) ----------
theme_els <- function() {
  theme_bw(base_size = 11) +
    theme(
      panel.grid = element_blank(),
      panel.border = element_rect(color = "black", size = 0.3),
      axis.text.x = element_text(angle = 30, hjust = 1),
      legend.position = "top",
      legend.title = element_blank()
    )
}

palette_els <- c(
  "MAE_CL" = "#264653",   # dark blue
  "MAE_TL" = "#7d7d7d"    # medium gray
)


# ---------- Generic MAE bar-plot helper ----------
plot_mae_group <- function(df, title_text) {
  
  df_long <- df %>%
    pivot_longer(
      cols = c(MAE_CL, MAE_TL),
      names_to = "metric",
      values_to = "value"
    ) %>%
    filter(!is.na(value))
  
  ggplot(
    df_long,
    aes(
      x = factor(display_label, levels = unique(display_label)),
      y = value,
      fill = metric
    )
  ) +
    geom_col(position = position_dodge(width = 0.75), width = 0.6) +
    scale_fill_manual(values = palette_els) +
    labs(
      title = title_text,
      x = "Model",
      y = "MAE (mm)"
    ) +
    theme_els()
}

# ============================================================
# GROUP I — BASELINES
# ============================================================
# ============================================================
# GROUP I — BASELINES  on pdf 
# ============================================================


baseline_groups <- c(
  "Baseline-2024-All",
  "Baseline-2024-Circ",
  "Baseline-2024-Square",
  "Baseline-2025-All"
)


g1_all <- classified_df %>%
  filter(precise_group %in% baseline_groups) %>%
  mutate(
    display_label = recode(
      precise_group,
      "Baseline-2024-All"    = "2024-All",
      "Baseline-2024-Circ"   = "2024-Circ",
      "Baseline-2024-Square" = "2024-Sq",
      "Baseline-2025-All"    = "2025-All"
    ),

    display_label = factor(
      display_label,
      levels = c("2024-All", "2024-Circ", "2024-Sq", "2025-All")
    )
  )

# ----- Group I, test on 2024 -----
g1_2024 <- g1_all %>%
  filter(test_year == 2024)

plot_g1_2024 <- plot_mae_group(
  g1_2024,
  "Group I — Full single-season baselines (Test 2024)"
)

plot_g1_2024

# ----- Group I, test on 2025 -----
g1_2025 <- g1_all %>%
  filter(test_year == 2025)

plot_g1_2025 <- plot_mae_group(
  g1_2025,
  "Group I — Full single-season baselines (Test 2025)"
)

plot_g1_2025

# ============================================================
# GROUP II — Cross-season TL + baseline in same plot
# Baseline MAE bars have different colors than CS models
# ============================================================

library(tidyverse)

# ---------- Theme ----------
theme_els <- function() {
  theme_bw(base_size = 11) +
    theme(
      panel.grid = element_blank(),
      panel.border = element_rect(color = "black", linewidth = 0.3),
      axis.text.x = element_text(angle = 30, hjust = 1),
      legend.position = "top",
      legend.title = element_blank()
    )
}

# ---------- Color palette ----------
# 4 distinct colors:
#   - 2 for cross-season models (MAE_CL, MAE_TL)
#   - 2 for baseline models (MAE_CL, MAE_TL)
palette_mae_type <- c(
  "MAE_CL.Cross-season model"       = "#264653",  # dark blue (CS carapace)
  "MAE_TL.Cross-season model"       = "#7d7d7d",  # gray (CS total)
  "MAE_CL.Baseline (All-ponds)"     = "#e76f51",  # orange-red (baseline carapace)
  "MAE_TL.Baseline (All-ponds)"     = "#f4a261"   # orange (baseline total)
)

labels_mae_type <- c(
  "MAE_CL.Cross-season model"       = "MAE – carapace (CS models)",
  "MAE_TL.Cross-season model"       = "MAE – total length (CS models)",
  "MAE_CL.Baseline (All-ponds)"     = "MAE – carapace (Baseline)",
  "MAE_TL.Baseline (All-ponds)"     = "MAE – total length (Baseline)"
)

# ============================================================
# 1. Cross-season (CS) models: average MAE per label & test_year
# ============================================================
cs_labeled <- classified_df %>%
  filter(str_detect(experiment_group, "CS")) %>%
  mutate(
    display_label = case_when(
      experiment_group == "CS-24→25 (All→All)"  ~ "CS 24→25 — All→All",
      experiment_group == "CS-24→25 (Circ→All)" ~ "CS 24→25 — Circ→All",
      experiment_group == "CS-25→24 (All→All)"  ~ "CS 25→24 — All→All",
      experiment_group == "CS-25→24 (All→Circ)" ~ "CS 25→24 — All→Circ",
      TRUE ~ experiment_group
    )
  ) %>%
  group_by(experiment_group, test_year, display_label) %>%
  summarise(
    MAE_CL = mean(MAE_CL, na.rm = TRUE),
    MAE_TL = mean(MAE_TL, na.rm = TRUE),
    .groups = "drop"
  )

# ============================================================
# 2. Baselines: 2024-All and 2025-All (all-ponds models)
# ============================================================
baseline_rows <- classified_df %>%
  filter(precise_group %in% c("Baseline-2024-All", "Baseline-2025-All")) %>%
  mutate(
    display_label = case_when(
      precise_group == "Baseline-2024-All" ~ "2024-All (Baseline)",
      precise_group == "Baseline-2025-All" ~ "2025-All (Baseline)",
      TRUE ~ precise_group
    )
  ) %>%
  group_by(precise_group, test_year, display_label) %>%
  summarise(
    MAE_CL = mean(MAE_CL, na.rm = TRUE),
    MAE_TL = mean(MAE_TL, na.rm = TRUE),
    .groups = "drop"
  )

# ============================================================
# 3. Plot function: MAE_CL + MAE_TL, with different colors for baseline
# ============================================================
plot_group2_mae_both <- function(year_selected) {
  
  # Cross-season models for this test year
  cs_year <- cs_labeled %>%
    filter(test_year == year_selected) %>%
    mutate(type = "Cross-season model")
  
  # Baseline for this test year
  base_year <- baseline_rows %>%
    filter(test_year == year_selected) %>%
    mutate(type = "Baseline (All-ponds)")
  
  # Combine CS + baseline
  combined <- bind_rows(cs_year, base_year)
  
  # Order: baseline first (if exists), then CS models
  labels_order <- c(base_year$display_label, cs_year$display_label) %>% unique()
  
  df_plot <- combined %>%
    pivot_longer(
      cols = c(MAE_CL, MAE_TL),
      names_to = "metric",
      values_to = "value"
    ) %>%
    filter(!is.na(value)) %>%
    mutate(
      display_label = factor(display_label, levels = labels_order),
      type          = factor(type, levels = c("Baseline (All-ponds)", "Cross-season model")),
      metric_type   = interaction(metric, type, lex.order = TRUE)
    )
  
  ggplot(
    df_plot,
    aes(
      x = display_label,
      y = value,
      fill = metric_type
    )
  ) +
    geom_col(
      position = position_dodge(width = 0.7),
      width = 0.6
    ) +
    scale_fill_manual(
      values = palette_mae_type,
      labels = labels_mae_type
    ) +
    labs(
      title = paste0("Group II — Cross-season + baseline (Test ", year_selected, ")"),
      x = "Model",
      y = "MAE (mm)"
    ) +
    theme_els()
}

# ============================================================
# 4. Generate the two plots (2024 and 2025 test sets)
# ============================================================
g2_mae_2024 <- plot_group2_mae_both(2024)
g2_mae_2025 <- plot_group2_mae_both(2025)

# Show them
g2_mae_2024
g2_mae_2025


# ============================================================
# B1 – show ALL experiments, highlight the best configuration
# ============================================================
# ============================================================
# GROUP B1 – Feature Extraction grid
# All experiments, best per (test_year, direction) highlighted
# ============================================================
# ============================================================
# GROUP 2 – B1 (Feature extraction) — both directions, both tests
# ============================================================
library(tidyverse)

# ------------------------------------------------------------
# 1. Parse B1 models and decode direction & freeze depth
# ------------------------------------------------------------
b1_raw <- classified_df %>%
  filter(experiment_group == "B1") %>%
  mutate(
    model_str = tolower(model),
    
    # Direction of transfer (source → target)
    direction = case_when(
      str_detect(model_str, "2024to2025") ~ "Source 24 \u2192 Target 25",
      str_detect(model_str, "2025to2024") ~ "Source 25 \u2192 Target 24",
      TRUE ~ "Unknown direction"
    ),
    
    # k-shot percentage (25 / 50 / 100)
    k_pct = k_shot,
    
    # Extract part after "freeze"
    #   ..._freeze02         -> "02"
    #   ..._freeze42         -> "42"
    #   ..._freeze82         -> "82"
    #   ..._freezeheadsonly2 -> "headsonly2"
    freeze_tag = str_match(model_str, "freeze(headsonly[0-9]*|[0-9]+)")[, 2],
    
    # Map to type
    freeze_type = case_when(
      is.na(freeze_tag) ~ NA_character_,
      str_detect(freeze_tag, "headsonly") ~ "heads-only",
      TRUE ~ freeze_tag
    ),
    
    # Decode numeric freeze depth:
    # "02" -> 0 backbone blocks frozen (full fine-tuning)
    # "42" -> 4 backbone blocks frozen
    # "82" -> 8 backbone blocks frozen
    # "2"  -> 2, etc.
    freeze_depth = case_when(
      freeze_type == "heads-only" ~ NA_real_,
      str_detect(freeze_type, "^[0-9]+$") ~ as.numeric(str_sub(freeze_type, 1, 1)),
      TRUE ~ NA_real_
    ),
    
    # Human-readable configuration label for x-axis
    config_label = case_when(
      !is.na(freeze_depth) ~ paste0("k=", k_pct, "%, freeze=", freeze_depth),
      freeze_type == "heads-only" ~ paste0("k=", k_pct, "%, heads-only"),
      TRUE ~ paste0("k=", k_pct, "%, freeze=?")
    )
  )

# ------------------------------------------------------------
# 2. Aggregate per configuration
#    (one row per: direction × test_year × k × freeze)
# ------------------------------------------------------------
b1_summary <- b1_raw %>%
  group_by(direction, test_year, config_label, k_pct, freeze_depth, freeze_type) %>%
  summarise(
    MAE_CL = mean(MAE_CL, na.rm = TRUE),
    MAE_TL = mean(MAE_TL, na.rm = TRUE),
    .groups = "drop"
  )

# ------------------------------------------------------------
# 3. Best configuration per (direction, test_year)
#    Criterion: minimal average of MAE_CL and MAE_TL
# ------------------------------------------------------------
b1_best <- b1_summary %>%
  mutate(avg_mae = (MAE_CL + MAE_TL) / 2) %>%
  filter(!is.na(avg_mae)) %>%
  group_by(direction, test_year) %>%
  slice_min(avg_mae, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  mutate(is_best = TRUE) %>%
  select(direction, test_year, config_label, is_best)

b1_plot_df <- b1_summary %>%
  left_join(b1_best, by = c("direction", "test_year", "config_label")) %>%
  mutate(is_best = replace_na(is_best, FALSE))

# ------------------------------------------------------------
# 4. Colors: base vs best
# ------------------------------------------------------------
cols_b1 <- c(
  "MAE_CL_base" = "#264653",  # dark blue
  "MAE_TL_base" = "#7d7d7d",  # gray
  "MAE_CL_best" = "#e76f51",  # orange
  "MAE_TL_best" = "#f4a261"   # light orange
)

# ------------------------------------------------------------
# 5. Helper: build plot for a given test year (both directions)
# ------------------------------------------------------------
plot_b1_test <- function(year_selected) {
  df <- b1_plot_df %>%
    filter(test_year == year_selected)
  
  if (nrow(df) == 0) {
    warning(paste("No rows for test_year =", year_selected))
    return(NULL)
  }
  
  # Order configs: by direction, then freeze depth, then k%
  config_levels <- df %>%
    arrange(direction, freeze_depth, k_pct, config_label) %>%
    pull(config_label) %>%
    unique()
  
  df_long <- df %>%
    pivot_longer(
      cols = c(MAE_CL, MAE_TL),
      names_to = "metric",
      values_to = "value"
    ) %>%
    filter(!is.na(value)) %>%
    mutate(
      role = if_else(is_best, "best", "base"),
      metric_role = paste(metric, role, sep = "_"),
      config_label = factor(config_label, levels = config_levels)
    )
  
  ggplot(
    df_long,
    aes(x = config_label, y = value, fill = metric_role)
  ) +
    geom_col(position = position_dodge(width = 0.8), width = 0.6) +
    facet_wrap(~ direction, scales = "free_x") +
    scale_fill_manual(
      values = cols_b1,
      breaks = c("MAE_CL_base", "MAE_TL_base", "MAE_CL_best", "MAE_TL_best"),
      labels = c(
        "MAE CL",
        "MAE TL",
        "MAE CL (best configuration)",
        "MAE TL (best configuration)"
      )
    ) +
    labs(
      title = paste0(
        "Group 2 – B1 (Feature extraction) — Test ", year_selected
      ),
      x = "Configuration (k-shot and freeze depth)",
      y = "MAE (mm)"
    ) +
    theme_bw(base_size = 11) +
    theme(
      axis.text.x = element_text(angle = 30, hjust = 1),
      legend.position = "top",
      legend.title = element_blank()
    )
}

# ------------------------------------------------------------
# 6. Build plots for both test years
# ------------------------------------------------------------
b1_plot_test_2024 <- plot_b1_test(2024)
b1_plot_test_2025 <- plot_b1_test(2025)

# View in RStudio
b1_plot_test_2024
b1_plot_test_2025


# ============================================================
# GROUP 2 – B2 (Fine-tuning) — both directions, both tests
# ============================================================
library(tidyverse)

# ------------------------------------------------------------
# 1. Parse B2 models and decode direction & freeze depth
# ------------------------------------------------------------
b2_raw <- classified_df %>%
  filter(experiment_group == "B2") %>%
  mutate(
    model_str = tolower(model),
    
    # Direction of transfer (source → target)
    direction = case_when(
      str_detect(model_str, "2024to2025") ~ "Source 24 \u2192 Target 25",
      str_detect(model_str, "2025to2024") ~ "Source 25 \u2192 Target 24",
      TRUE ~ "Unknown direction"
    ),
    
    # k-shot percentage (25 / 50 / 100)
    k_pct = k_shot,
    
    # For B2 we can use the 'freeze' column computed earlier
    freeze_depth = freeze,
    
    # Human-readable configuration label for x-axis
    config_label = case_when(
      !is.na(freeze_depth) & !is.na(k_pct) ~ paste0("k=", k_pct, "%, freeze=", freeze_depth),
      !is.na(k_pct) ~ paste0("k=", k_pct, "%"),
      TRUE ~ "k=?, freeze=?"
    )
  )

# ------------------------------------------------------------
# 2. Aggregate per configuration
#    (one row per: direction × test_year × k × freeze)
# ------------------------------------------------------------
b2_summary <- b2_raw %>%
  group_by(direction, test_year, config_label, k_pct, freeze_depth) %>%
  summarise(
    MAE_CL = mean(MAE_CL, na.rm = TRUE),
    MAE_TL = mean(MAE_TL, na.rm = TRUE),
    .groups = "drop"
  )

# ------------------------------------------------------------
# 3. Best configuration per (direction, test_year)
#    Criterion: minimal average of MAE_CL and MAE_TL
# ------------------------------------------------------------
b2_best <- b2_summary %>%
  mutate(avg_mae = (MAE_CL + MAE_TL) / 2) %>%
  filter(!is.na(avg_mae)) %>%
  group_by(direction, test_year) %>%
  slice_min(avg_mae, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  mutate(is_best = TRUE) %>%
  select(direction, test_year, config_label, is_best)

b2_plot_df <- b2_summary %>%
  left_join(b2_best, by = c("direction", "test_year", "config_label")) %>%
  mutate(is_best = replace_na(is_best, FALSE))

# ------------------------------------------------------------
# 4. Colors: base vs best (same style as B1)
# ------------------------------------------------------------
cols_b2 <- c(
  "MAE_CL_base" = "#264653",  # dark blue
  "MAE_TL_base" = "#7d7d7d",  # gray
  "MAE_CL_best" = "#e76f51",  # orange
  "MAE_TL_best" = "#f4a261"   # light orange
)

# ------------------------------------------------------------
# 5. Helper: build plot for a given test year (both directions)
# ------------------------------------------------------------
plot_b2_test <- function(year_selected) {
  df <- b2_plot_df %>%
    filter(test_year == year_selected)
  
  if (nrow(df) == 0) {
    warning(paste("No rows for test_year =", year_selected))
    return(NULL)
  }
  
  # Order configs: by direction, then freeze depth, then k%
  config_levels <- df %>%
    arrange(direction, freeze_depth, k_pct, config_label) %>%
    pull(config_label) %>%
    unique()
  
  df_long <- df %>%
    pivot_longer(
      cols = c(MAE_CL, MAE_TL),
      names_to = "metric",
      values_to = "value"
    ) %>%
    filter(!is.na(value)) %>%
    mutate(
      role = if_else(is_best, "best", "base"),
      metric_role = paste(metric, role, sep = "_"),
      config_label = factor(config_label, levels = config_levels)
    )
  
  ggplot(
    df_long,
    aes(x = config_label, y = value, fill = metric_role)
  ) +
    geom_col(position = position_dodge(width = 0.8), width = 0.6) +
    facet_wrap(~ direction, scales = "free_x") +
    scale_fill_manual(
      values = cols_b2,
      breaks = c("MAE_CL_base", "MAE_TL_base", "MAE_CL_best", "MAE_TL_best"),
      labels = c(
        "MAE CL",
        "MAE TL",
        "MAE CL (best configuration)",
        "MAE TL (best configuration)"
      )
    ) +
    labs(
      title = paste0(
        "Group 2 – B2 (Fine-tuning) — Test ", year_selected
      ),
      x = "Configuration (k-shot and freeze depth)",
      y = "MAE (mm)"
    ) +
    theme_bw(base_size = 11) +
    theme(
      axis.text.x = element_text(angle = 30, hjust = 1),
      legend.position = "top",
      legend.title = element_blank()
    )
}

# ------------------------------------------------------------
# 6. Build plots for both test years
# ------------------------------------------------------------
b2_plot_test_2024 <- plot_b2_test(2024)
b2_plot_test_2025 <- plot_b2_test(2025)

# View in RStudio
b2_plot_test_2024
b2_plot_test_2025



# ============================================================
# GROUP 2 – B3 (Multi-stage transfer) WITH CORRECT BASELINE BAR
# ============================================================
library(tidyverse)

# ------------------------------------------------------------
# 1. Parse B3 models: decode direction, k-shot, freeze depth
# ------------------------------------------------------------
b3_raw <- classified_df %>%
  filter(experiment_group == "B3") %>%
  mutate(
    model_str = tolower(model),
    
    # Transfer direction
    direction = case_when(
      str_detect(model_str, "2024to2025") ~ "Source 24 → Target 25",
      str_detect(model_str, "2025to2024") ~ "Source 25 → Target 24",
      TRUE ~ "Unknown direction"
    ),
    
    # k-shot
    k_pct = k_shot,
    
    # freeze depth
    freeze_depth = freeze,
    
    # Clean label
    config_label = paste0("k=", k_pct, "%, freeze=", freeze_depth)
  )

# ------------------------------------------------------------
# 2. Summaries per configuration (combine identical runs)
# ------------------------------------------------------------
b3_summary <- b3_raw %>%
  group_by(direction, test_year, config_label, k_pct, freeze_depth) %>%
  summarise(
    MAE_CL = mean(MAE_CL, na.rm = TRUE),
    MAE_TL = mean(MAE_TL, na.rm = TRUE),
    .groups = "drop"
  )

# ------------------------------------------------------------
# 3. Best configuration = lowest average of MAEs
# ------------------------------------------------------------
b3_best <- b3_summary %>%
  mutate(avg_mae = (MAE_CL + MAE_TL)/2) %>%
  group_by(direction, test_year) %>%
  slice_min(avg_mae, with_ties = FALSE) %>%
  ungroup() %>%
  mutate(is_best = TRUE)

b3_plot_df <- b3_summary %>%
  left_join(
    b3_best %>% select(direction, test_year, config_label, is_best),
    by = c("direction", "test_year", "config_label")
  ) %>%
  mutate(is_best = replace_na(is_best, FALSE))

# ------------------------------------------------------------
# 4. Extract only the correct baseline per test-year
# ------------------------------------------------------------
baseline_values <- classified_df %>%
  filter(precise_group %in% c("Baseline-2024-All", "Baseline-2025-All")) %>%
  mutate(
    direction = "Baseline",
    config_label = case_when(
      precise_group == "Baseline-2024-All" ~ "Baseline 2024",
      precise_group == "Baseline-2025-All" ~ "Baseline 2025"
    ),
    is_best = FALSE   # <<------ FIX: always logical
  ) %>%
  select(direction, test_year, config_label, MAE_CL, MAE_TL, is_best)

# ------------------------------------------------------------
# 5. Combine but only keep baseline for matching test-year
# ------------------------------------------------------------
b3_with_baseline <- bind_rows(
  b3_plot_df,
  baseline_values
)

# ------------------------------------------------------------
# 6. Colors
# ------------------------------------------------------------
cols_b3 <- c(
  # Base configs (subtle grays)
  "MAE_CL_base"     = "#4B4B4B",
  "MAE_TL_base"     = "#B0B0B0",
  
  # Best configs (highlight orange)
  "MAE_CL_best"     = "#D95F02",
  "MAE_TL_best"     = "#FDAE6B",
  
  # Baseline (professional blue-gray shades)
  "MAE_CL_baseline" = "#3C5878",
  "MAE_TL_baseline" = "#9EAEC3"
)

# ------------------------------------------------------------
# 7. Plot function
# ------------------------------------------------------------
plot_b3_final <- function(year_selected) {
  
  df <- b3_with_baseline %>% filter(test_year == year_selected)
  if (nrow(df) == 0) return(NULL)
  
  df_long <- df %>%
    pivot_longer(
      cols = c(MAE_CL, MAE_TL),
      names_to = "metric",
      values_to = "value"
    ) %>%
    mutate(
      role = case_when(
        direction == "Baseline" ~ "baseline",
        is_best == TRUE ~ "best",
        TRUE ~ "base"
      ),
      metric_role = paste(metric, role, sep = "_")
    ) %>%
    # keep only the correct baseline
    filter(!(direction == "Baseline" & config_label == "Baseline 2024" & year_selected != 2024)) %>%
    filter(!(direction == "Baseline" & config_label == "Baseline 2025" & year_selected != 2025))
  
  # Order: baseline first
  ordered_labels <- df_long %>%
    arrange(direction == "Baseline", freeze_depth, k_pct) %>%
    pull(config_label) %>%
    unique()
  
  df_long$config_label <- factor(df_long$config_label, levels = ordered_labels)
  
  ggplot(df_long,
         aes(x = config_label, y = value, fill = metric_role)) +
    geom_col(position = position_dodge(width = 0.75), width = 0.6) +
    facet_wrap(~ direction, scales = "free_x") +
    scale_fill_manual(values = cols_b3) +
    labs(
      title = paste0("Group 2 – B3 Multi-stage transfer ", year_selected),
      x = "Configuration (k-shot & freeze)",
      y = "MAE (mm)"
    ) +
    theme_bw(base_size = 11) +
    theme(
      axis.text.x = element_text(angle = 30, hjust = 1),
      legend.position = "top",
      legend.title = element_blank()
    )
}

# ------------------------------------------------------------
# 8. Produce final plots
# ------------------------------------------------------------
b3_plot_test_2024 <- plot_b3_final(2024)
b3_plot_test_2025 <- plot_b3_final(2025)

b3_plot_test_2024
b3_plot_test_2025

library(tidyverse)

# ============================================================
# 1. Filter only B2 experiments
# ============================================================
b2 <- classified_df %>%
  filter(experiment_group == "B2") %>%
  mutate(
    direction = case_when(
      str_detect(tolower(model), "2024to2025") ~ "Source 24 → Target 25",
      str_detect(tolower(model), "2025to2024") ~ "Source 25 → Target 24",
      TRUE ~ "Unknown"
    )
  )

# ============================================================
# 2. Average MAE per freeze depth
# ============================================================
b2_freeze <- b2 %>%
  group_by(test_year, direction, freeze, k_shot) %>%
  summarise(
    MAE_CL = mean(MAE_CL, na.rm = TRUE),
    MAE_TL = mean(MAE_TL, na.rm = TRUE),
    .groups = "drop"
  )

# ============================================================
# 3. Long format for plotting
# ============================================================
b2_freeze_long <- b2_freeze %>%
  pivot_longer(
    cols = c(MAE_CL, MAE_TL),
    names_to = "metric",
    values_to = "value"
  )

# ============================================================
# 4. Plot — MAE vs Freeze Depth (per direction + test_year)
# ============================================================
plot_b2_freeze <- ggplot(
  b2_freeze_long,
  aes(x = freeze, y = value, color = metric)
) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  facet_grid(test_year ~ direction) +
  scale_color_manual(
    values = c("MAE_CL" = "#264653", "MAE_TL" = "#7d7d7d"),
    labels = c("MAE (Carapace)", "MAE (Total length)")
  ) +
  labs(
    title = "B2 – Trend: MAE vs Freeze Depth",
    x = "Freeze Depth",
    y = "MAE (mm)",
    color = "Metric"
  ) +
  theme_bw(base_size = 12)

plot_b2_freeze
# ============================================================
# 1. Average MAE per k-shot
# ============================================================
b2_k <- b2 %>%
  group_by(test_year, direction, k_shot, freeze) %>%
  summarise(
    MAE_CL = mean(MAE_CL, na.rm = TRUE),
    MAE_TL = mean(MAE_TL, na.rm = TRUE),
    .groups = "drop"
  )

# ============================================================
# 2. Long format for plotting
# ============================================================
b2_k_long <- b2_k %>%
  pivot_longer(
    cols = c(MAE_CL, MAE_TL),
    names_to = "metric",
    values_to = "value"
  )

# ============================================================
# 3. Plot — MAE vs k-shot (per direction + test_year)
# ============================================================
plot_b2_k <- ggplot(
  b2_k_long,
  aes(x = k_shot, y = value, color = metric)
) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  facet_grid(test_year ~ direction) +
  scale_color_manual(
    values = c("MAE_CL" = "#264653", "MAE_TL" = "#7d7d7d"),
    labels = c("MAE (Carapace)", "MAE (Total length)")
  ) +
  labs(
    title = "B2 – Trend: MAE vs k-shot (%)",
    x = "k-shot (%)",
    y = "MAE (mm)",
    color = "Metric"
  ) +
  theme_bw(base_size = 12)

plot_b2_k
