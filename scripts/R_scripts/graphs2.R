# ============================================================
# FULL ACADEMIC ANALYSIS PIPELINE: PRAWN MORPHOMETRICS
# ============================================================

# ------------------------------------------------------------
# 1. LIBRARIES & VISUAL SETUP
# ------------------------------------------------------------
library(tidyverse)
library(readxl)
library(janitor)
library(stringr)
library(writexl)
# Define the visual theme for the paper
theme_paper <- theme_bw(base_size = 12) +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    legend.position = "top",
    legend.title = element_blank(),
    axis.text = element_text(color = "black"),
    strip.background = element_rect(fill = "#f0f0f0"),
    plot.title = element_text(face = "bold", size = 13, hjust = 0), # Left align title
    plot.subtitle = element_text(size = 11, color = "gray30")
  )

# Define colors and labels
colors_mae <- c("MAE_CL" = "#264653", "MAE_TL" = "#e76f51") 
labels_mae <- c("MAE_CL" = "Carapace Length (CL)", "MAE_TL" = "Total Length (TL)")

# ------------------------------------------------------------
# 2. DATA LOADING & CLEANING FUNCTIONS
# ------------------------------------------------------------

clean_num <- function(x) {
  x <- as.character(x)
  x <- str_replace_all(x, "[^0-9\\.\\-]", "")
  suppressWarnings(as.numeric(x))
}

infer_year_from_path <- function(path) {
  if (str_detect(path, "test_on_2024")) return(2024L)
  if (str_detect(path, "test_on_2025")) return(2025L)
  NA_integer_
}

process_file <- function(path) {
  year <- infer_year_from_path(path)
  # Read sheet "ALL_MODELS"
  df <- suppressMessages(read_excel(path, sheet = "ALL_MODELS")) %>% clean_names()
  
  # Normalize columns based on year format
  if (year == 2024) {
    if (!"model" %in% names(df)) return(tibble())
    df_2024 <- df %>%
      mutate(metric = str_squish(as.character(metric)), 
             model = str_squish(as.character(model)), 
             combined = clean_num(combined)) %>%
      mutate(metric_std = case_when(
        str_detect(metric, "MAE carapace") ~ "MAE_CL",
        str_detect(metric, "MAE total")    ~ "MAE_TL",
        str_detect(metric, "(?i)detection|rate|recall") & str_detect(metric, "carapace") ~ "DR_CL",
        str_detect(metric, "(?i)detection|rate|recall") & str_detect(metric, "total")    ~ "DR_TL",
        TRUE ~ NA_character_
      )) %>%
      filter(!is.na(metric_std)) %>%
      select(model, metric_std, combined) %>%
      pivot_wider(names_from = metric_std, values_from = combined) %>%
      mutate(test_year = year)
    return(df_2024)
  }
  
  if (year == 2025) {
    # Dynamically find column names
    col_mae_cl <- names(df)[str_detect(names(df), "mae.*carapace")]
    col_mae_tl <- names(df)[str_detect(names(df), "mae.*total")]
    col_dr_cl  <- names(df)[str_detect(names(df), "detection.*carapace|recall.*carapace")]
    col_dr_tl  <- names(df)[str_detect(names(df), "detection.*total|recall.*total")]
    
    df_2025 <- df %>%
      mutate(model = str_squish(as.character(model))) %>%
      transmute(
        model,
        MAE_CL = if (length(col_mae_cl) > 0) clean_num(.data[[col_mae_cl[1]]]) else NA_real_,
        MAE_TL = if (length(col_mae_tl) > 0) clean_num(.data[[col_mae_tl[1]]]) else NA_real_,
        DR_CL  = if (length(col_dr_cl) > 0)  clean_num(.data[[col_dr_cl[1]]])  else NA_real_,
        DR_TL  = if (length(col_dr_tl) > 0)  clean_num(.data[[col_dr_tl[1]]])  else NA_real_,
        test_year = year
      )
    return(df_2025)
  }
  return(tibble())
}

process_file <- function(path) {
  year <- infer_year_from_path(path)
  # Read sheet "ALL_MODELS"
  df <- suppressMessages(read_excel(path, sheet = "ALL_MODELS")) %>% clean_names()
  
  # Normalize columns based on year format
  if (year == 2024) {
    if (!"model" %in% names(df)) return(tibble())
    
    df_2024 <- df %>%
      mutate(metric = str_squish(as.character(metric)), 
             model = str_squish(as.character(model)), 
             combined = clean_num(combined)) %>%
      mutate(metric_std = case_when(
        # --- תיקון גמישות: מחפש 'carapa' במקום 'carapace' ---
        
        # 1. זיהוי MAE
        str_detect(metric, "(?i)mae") & str_detect(metric, "(?i)carapa") ~ "MAE_CL",
        str_detect(metric, "(?i)mae") & str_detect(metric, "(?i)total")  ~ "MAE_TL",
        
        # 2. זיהוי Detection (תופס גם rate, frequency, recall)
        str_detect(metric, "(?i)(detect|rate|recall|freq)") & str_detect(metric, "(?i)carapa") ~ "Det_CL",
        str_detect(metric, "(?i)(detect|rate|recall|freq)") & str_detect(metric, "(?i)total")  ~ "Det_TL",
        
        TRUE ~ NA_character_
      )) %>%
      filter(!is.na(metric_std)) %>%
      select(model, metric_std, combined) %>%
      pivot_wider(names_from = metric_std, values_from = combined) %>%
      mutate(test_year = year)
    
    return(df_2024)
  }
  
  # ... (המשך הקוד לשנת 2025 נשאר אותו דבר)
  if (year == 2025) {
    # Dynamically find column names
    col_mae_cl <- names(df)[str_detect(names(df), "mae.*carapace")]
    col_mae_tl <- names(df)[str_detect(names(df), "mae.*total")]
    col_det_cl <- names(df)[str_detect(names(df), "detection.*carapace|recall.*carapace")]
    col_det_tl <- names(df)[str_detect(names(df), "detection.*total|recall.*total")]
    
    df_2025 <- df %>%
      mutate(model = str_squish(as.character(model))) %>%
      transmute(
        model,
        MAE_CL = if (length(col_mae_cl) > 0) clean_num(.data[[col_mae_cl[1]]]) else NA_real_,
        MAE_TL = if (length(col_mae_tl) > 0) clean_num(.data[[col_mae_tl[1]]]) else NA_real_,
        Det_CL = if (length(col_det_cl) > 0) clean_num(.data[[col_det_cl[1]]]) else NA_real_,
        Det_TL = if (length(col_det_tl) > 0) clean_num(.data[[col_det_tl[1]]]) else NA_real_,
        test_year = year
      )
    return(df_2025)
  }
  return(tibble())
}
# ------------------------------------------------------------
# 3. LOAD RAW DATA
# ------------------------------------------------------------
# Define paths
dir_tf  <- "/Users/taircarmon/Desktop/Uni-Projects/prawn-size-project/scripts/eval/outputs_TF"
dir_out <- "/Users/taircarmon/Desktop/Uni-Projects/prawn-size-project/scripts/eval/outputs"

files <- c(
  list.files(dir_tf,  pattern = "all_models.*\\.xlsx$", recursive = TRUE, full.names = TRUE),
  list.files(dir_out, pattern = "all_models.*\\.xlsx$", recursive = TRUE, full.names = TRUE)
)

cat("Loading", length(files), "files...\n")
raw_data <- map_df(files, process_file)

# ------------------------------------------------------------
# 4. DATA MAPPING & FILTERING
# ------------------------------------------------------------
# ------------------------------------------------------------
# 4. DATA MAPPING & FILTERING (WITH BUG FIX)
# ------------------------------------------------------------

clean_data <- raw_data %>%
  mutate(
    m = tolower(model),
    
    # --- A. ACADEMIC NAMING ---
    Paper_Name = case_when(
      str_detect(m, "^all-ponds") ~ "Base-S24",
      str_detect(m, "^circulars") ~ "Base-S24-Circ",
      str_detect(m, "^square")    ~ "Base-S24-Sq",
      str_detect(m, "train_on_2025_all") & !str_detect(m, "2024") ~ "Base-S25",
      str_detect(m, "train_on_2024_2025_all")      ~ "Joint-S24+S25",
      str_detect(m, "train_on_2024_2025_circular") ~ "Joint-Circ+S25",
      str_detect(m, "circular_to_square") ~ "TL-Circ->Sq",
      str_detect(m, "square_to_circular") ~ "TL-Sq->Circ",
      str_detect(m, "2024all_to_2025all")  ~ "TL-S24->S25 (All->All)",
      str_detect(m, "2024circ_to_2025all") ~ "TL-S24->S25 (Circ->All)",
      str_detect(m, "2025all_to_2024all")  ~ "TL-S25->S24 (All->All)",
      str_detect(m, "2025all_to_2024circ") ~ "TL-S25->S24 (All->Circ)",
      TRUE ~ "Experimental"
    ),
    
    # --- B. PRIORITY ---
    priority = case_when(
      str_detect(m, "best.pt") ~ 2,
      str_detect(m, "artifact") ~ 1,
      TRUE ~ 0 
    )
  ) %>%
  
  # --- C. REMOVE DUPLICATES ---
  group_by(test_year, Paper_Name, 
           group_id = ifelse(Paper_Name == "Experimental", m, Paper_Name)) %>%
  slice_max(order_by = priority, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  
  # --- D. PARAMETERS (WITH FIX FOR 82/42) ---
  mutate(
    k_pct = as.numeric(str_extract(m, "(?<=k)[0-9]+")),
    
    # 1. Extract the number as is (e.g., 82)
    raw_freeze = as.numeric(str_extract(m, "(?<=freeze)[0-9]+")),
    
    # 2. Apply the correction logic
    freeze = case_when(
      raw_freeze == 82 ~ 8,   # Fix: 82 becomes 8
      raw_freeze == 42 ~ 4,   # Fix: 42 becomes 4
      TRUE ~ raw_freeze       # Keep others (e.g., 10, 0)
    ),
    
    direction = case_when(
      str_detect(m, "2024.*to.*2025") ~ "S24->S25",
      str_detect(m, "2025.*to.*2024") ~ "S25->S24",
      TRUE ~ "None"
    ),
    
    model_type = case_when(
      Paper_Name %in% c("Base-S24", "Base-S25", "Base-S24-Circ", "Base-S24-Sq") ~ "Baseline",
      str_detect(m, "b3") ~ "B3",
      str_detect(m, "b1") ~ "B1", 
      str_detect(m, "b2") | (Paper_Name == "Experimental" & !is.na(k_pct)) ~ "B2", 
      TRUE ~ "Other"
    ),
    Avg_Error = rowMeans(select(., MAE_CL, MAE_TL), na.rm = TRUE),
    Avg_DR    = rowMeans(select(., DR_CL, DR_TL), na.rm = TRUE)
  )

# ------------------------------------------------------------
# 5. HELPER: PRINT NICE TABLE
# ------------------------------------------------------------
print_nice_table <- function(data_used, title) {
  cat("\n============================================================\n")
  cat(" TABLE: ", title, "\n")
  cat("============================================================\n")
  
  table_out <- data_used %>%
    select(Label = plot_label, MAE_CL, MAE_TL, Full_Model_File = model) %>%
    mutate(
      MAE_CL = round(MAE_CL, 4),
      MAE_TL = round(MAE_TL, 4)
    ) %>%
    arrange(Label)
  
  print(as.data.frame(table_out), right = FALSE)
  cat("============================================================\n\n")
}

# ------------------------------------------------------------
# 6. ANALYSIS: FIGURE 6 (B2 TL-OPTIM)
# ------------------------------------------------------------

# Find Global B2 Winner
b2_winner <- clean_data %>%
  filter(model_type == "B2") %>%
  arrange(desc(Avg_DR), Avg_Error) %>%
  slice(1)

best_k_b2  <- b2_winner$k_pct
best_fr_b2 <- b2_winner$freeze

# Prepare Data
df_fig6 <- clean_data %>%
  mutate(
    plot_label = case_when(
      Paper_Name == "Base-S24" ~ "Base-2024 (all-ponds)",
      Paper_Name == "Base-S25" ~ "Base-2025 (all-2025)",
      model_type == "B2" & k_pct == best_k_b2 & freeze == best_fr_b2 ~ paste0("TL-Optim (k=", best_k_b2, ", fr=", best_fr_b2, ")"),
      TRUE ~ NA_character_
    )
  ) %>%
  filter(!is.na(plot_label))

plot_b2_ordered <- function(target_year, dir_regex, fig_number) {
  
  df_plot <- df_fig6 %>%
    filter(test_year == target_year) %>%
    filter(Paper_Name %in% c("Base-S24", "Base-S25") | 
             (str_detect(plot_label, "TL-Optim") & str_detect(m, dir_regex)))
  
  # PRINT TABLE
  print_nice_table(df_plot, paste("Figure", fig_number, "(B2 TL-Optim) - Target", target_year))
  
  # PLOT
  long_dat <- df_plot %>%
    pivot_longer(cols = c(MAE_CL, MAE_TL), names_to = "Metric", values_to = "Error")
  
  # FORCE ORDER
  b2_label <- unique(long_dat$plot_label[str_detect(long_dat$plot_label, "TL-Optim")])
  long_dat$plot_label <- factor(long_dat$plot_label, 
                                levels = c("Base-2024 (all-ponds)", "Base-2025 (all-2025)", b2_label))
  
  ggplot(long_dat, aes(x = plot_label, y = Error, fill = Metric)) +
    geom_col(position = position_dodge(0.8), width = 0.7) +
    geom_text(aes(label = round(Error, 2)), position = position_dodge(0.8), vjust = -0.5, fontface = "bold") +
    scale_fill_manual(values = colors_mae, labels = labels_mae) +
    # --- FIXES HERE ---
    # 1. Expand Y axis to prevent clipping at the top (adds 15% space at top)
    scale_y_continuous(expand = expansion(mult = c(0.05, 0.15))) +
    # 2. Better Academic Titles
    labs(title = paste0("Figure ", fig_number, ": Optimized Transfer Learning (TL-Optim) Performance"),
         subtitle = paste("Comparison of Mean Absolute Error on", target_year, "Test Set"),
         y = "Mean Absolute Error (mm)", 
         x = NULL) +
    theme_paper
}

cat("\n--- B2 PLOTS ---\n")
print(plot_b2_ordered(2025, "2024.*to.*2025", "6a"))
print(plot_b2_ordered(2024, "2025.*to.*2024", "6b"))

# ------------------------------------------------------------
# 7. ANALYSIS: FIGURE 6 (B3 MULTI-STAGE) - FIXED
# ------------------------------------------------------------

plot_b3_fixed <- function(target_year, fig_number) {
  
  # 1. Find Local Winner
  b3_local_winner <- clean_data %>%
    filter(test_year == target_year) %>%
    filter(model_type == "B3") %>%
    arrange(desc(Avg_DR), Avg_Error) %>%
    slice(1)
  
  if (nrow(b3_local_winner) == 0) {
    cat("\nWARNING: No B3 models found for test year", target_year, "\n")
    return(NULL)
  }
  
  best_name <- b3_local_winner$model
  best_k    <- b3_local_winner$k_pct
  best_fr   <- b3_local_winner$freeze
  
  cat("\nBest B3 for Target", target_year, "is:", best_name, "\n")
  
  # 2. Prepare Plot Data
  df_plot <- clean_data %>%
    filter(test_year == target_year) %>%
    mutate(
      plot_label = case_when(
        Paper_Name == "Base-S24" ~ "Base-2024 (all-ponds)",
        Paper_Name == "Base-S25" ~ "Base-2025 (all-2025)",
        model == best_name ~ paste0("B3 Multi-Stage (k=", best_k, ", fr=", best_fr, ")"),
        TRUE ~ NA_character_
      )
    ) %>%
    filter(!is.na(plot_label))
  
  # 3. PRINT TABLE
  print_nice_table(df_plot, paste("Figure", fig_number, "(B3 Multi-Stage) - Target", target_year))
  
  # 4. PLOT
  long_dat <- df_plot %>%
    pivot_longer(cols = c(MAE_CL, MAE_TL), names_to = "Metric", values_to = "Error")
  
  # FORCE ORDER
  b3_label <- unique(long_dat$plot_label[str_detect(long_dat$plot_label, "B3")])
  long_dat$plot_label <- factor(long_dat$plot_label, 
                                levels = c("Base-2024 (all-ponds)", "Base-2025 (all-2025)", b3_label))
  
  ggplot(long_dat, aes(x = plot_label, y = Error, fill = Metric)) +
    geom_col(position = position_dodge(0.8), width = 0.7) +
    geom_text(aes(label = round(Error, 2)), position = position_dodge(0.8), vjust = -0.5, fontface = "bold") +
    scale_fill_manual(values = colors_mae, labels = labels_mae) +
    # --- FIXES HERE ---
    # 1. Expand Y axis to prevent clipping
    scale_y_continuous(expand = expansion(mult = c(0.05, 0.15))) +
    # 2. Better Academic Titles
    labs(title = paste0("Figure ", fig_number, ": Multi-Stage Transfer Learning (B3) Performance"),
         subtitle = paste("Comparison of Mean Absolute Error on", target_year, "Test Set "),
         y = "Mean Absolute Error (mm)", 
         x = NULL) +
    theme_paper
}

cat("\n--- B3 PLOTS ---\n")
print(plot_b3_fixed(2025, "6c"))
print(plot_b3_fixed(2024, "6d"))

# ------------------------------------------------------------
# 8. ANALYSIS: FIGURES 7 & 8 (GRID ANALYSIS)
# ------------------------------------------------------------

# Prepare Grid Data (B2 only)
df_grid <- clean_data %>%
  filter(model_type == "B2") %>%
  filter(!is.na(freeze), !is.na(k_pct), direction != "None") %>%
  mutate(
    direction_nice = case_when(
      direction == "S24->S25" ~ "Transfer: 2024 to 2025",
      direction == "S25->S24" ~ "Transfer: 2025 to 2024"
    ),
    test_year_nice = paste("Test Year:", test_year)
  )

# --- FIGURE 7: FREEZE DEPTH ---
cat("\n--- FIGURE 7 DATA (Aggregated by Freeze) ---\n")
df_p7 <- df_grid %>%
  group_by(test_year_nice, direction_nice, freeze) %>%
  summarise(MAE_CL = mean(MAE_CL, na.rm=T), MAE_TL = mean(MAE_TL, na.rm=T), .groups="drop")

print(as.data.frame(df_p7), right = FALSE)

long_p7 <- df_p7 %>% pivot_longer(cols = c(MAE_CL, MAE_TL), names_to = "Metric", values_to = "Error")

p7 <- ggplot(long_p7, aes(x = freeze, y = Error, color = Metric)) +
  geom_line(linewidth = 1.2) + geom_point(size = 3) +
  facet_grid(test_year_nice ~ direction_nice) +
  # --- Better Titles ---
  labs(title = "Figure 7: Impact of Freeze Depth on Transfer Performance", 
       subtitle = "Mean Absolute Error across varying frozen backbone layers",
       x = "Number of Frozen Layers (Freeze Depth)", 
       y = "Mean Absolute Error (mm)") +
  scale_color_manual(values = colors_mae, labels = labels_mae) +
  theme_paper
print(p7)

# --- FIGURE 8: DATA EFFICIENCY ---
cat("\n--- FIGURE 8 DATA (Best config per Data %) ---\n")
df_p8 <- df_grid %>%
  group_by(test_year_nice, direction_nice, k_pct) %>%
  slice_min(Avg_Error, n=1) %>%
  select(test_year_nice, direction_nice, k_pct, freeze, MAE_CL, MAE_TL)

print(as.data.frame(df_p8), right = FALSE)

long_p8 <- df_p8 %>% pivot_longer(cols = c(MAE_CL, MAE_TL), names_to = "Metric", values_to = "Error")

p8 <- ggplot(long_p8, aes(x = k_pct, y = Error, color = Metric)) +
  geom_line(linewidth = 1.2) + geom_point(size = 3) +
  facet_grid(test_year_nice ~ direction_nice) +
  # --- Better Titles ---
  labs(title = "Figure 8: Data Efficiency in Transfer Learning", 
       subtitle = "Lowest achieved MAE vs. Percentage of Target Data Used",
       x = "Percentage of Target Data Used (k%)", 
       y = "Mean Absolute Error (mm)") +
  scale_color_manual(values = colors_mae, labels = labels_mae) +
  theme_paper
print(p8)

cat("\nAnalysis Complete.\n")
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
library(tidyverse)
library(ggplot2)

# ============================================================
# FINAL PLOT: No Cutting (Expanded Y-Axis)
# ============================================================

# 1. עיבוד נתונים (ללא שינוי)
df_averaged <- df_truth %>%
  group_by(Scenario_Label, k_pct) %>%
  summarise(
    MAE_CL = mean(MAE_CL, na.rm = TRUE),
    MAE_TL = mean(MAE_TL, na.rm = TRUE),
    DR_CL  = mean(DR_CL, na.rm = TRUE),
    DR_TL  = mean(DR_TL, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  pivot_longer(cols = c(MAE_CL, MAE_TL, DR_CL, DR_TL), 
               names_to = "Metric_Raw", values_to = "Value") %>%
  mutate(
    Metric_Type = factor(ifelse(str_detect(Metric_Raw, "MAE"), 
                                "Error (MAE)", "Detection Rate (%)"),
                         levels = c("Error (MAE)", "Detection Rate (%)")),
    Body_Part = ifelse(str_detect(Metric_Raw, "_CL"), "Carapace Length", "Total Length"),
    Value = ifelse(str_detect(Metric_Raw, "DR") & Value <= 1, Value * 100, Value)
  )

# 2. הציור המתוקן
ggplot(df_averaged, aes(x = k_pct, y = Value, color = Body_Part, group = Body_Part)) +
  
  # --- קו סף לזיהוי ---
  geom_hline(data = filter(df_averaged, str_detect(Metric_Type, "Detection")),
             aes(yintercept = 90), linetype = "dashed", color = "red", alpha = 0.4) +
  
  # --- קווים ונקודות ---
  geom_line(linewidth = 1.2) +
  geom_point(size = 3.5) +
  
  # --- תוויות מספרים (עם רקע לבן) ---
  geom_label(aes(label = round(Value, 1), 
                 vjust = ifelse(Body_Part == "Total Length", -0.4, 1.4)), 
             fill = "white", label.size = NA, alpha = 0.85,
             fontface = "bold", size = 3.5, show.legend = FALSE) +
  
  # --- הגריד ---
  facet_grid(Metric_Type ~ Scenario_Label, scales = "free_y", switch = "y") +
  
  # --- צירים וצבעים ---
  scale_x_continuous(breaks = c(25, 50, 100), labels = c("25%", "50%", "100%")) +
  scale_color_manual(values = c("Carapace Length" = "#264653", "Total Length" = "#e76f51")) +
  
  # --- כותרות ---
  labs(
    title = "Accuracy vs. Reliability Trade-off",
    subtitle = "Averaged across freeze depths. Dashed line = 90% Detection Threshold.",
    x = "Percentage of Target Data Used (k%)",
    y = NULL, 
    color = NULL
  ) +
  
  # --- התיקון הקריטי 1: הרחבה פנימית ---
  # הוספתי 30% רווח (0.3) למעלה ולמטה. זה המון מקום, המספרים לא יגעו בקצה.
  scale_y_continuous(expand = expansion(mult = c(0.3, 0.3))) +
  
  # --- התיקון הקריטי 2: ביטול חיתוך ---
  # מאפשר לצייר מחוץ לגבולות הפאנל
  coord_cartesian(clip = "off") +
  
  theme_bw(base_size = 14) +
  theme(
    strip.background = element_rect(fill = "#f0f0f0"),
    strip.text = element_text(face = "bold", size = 11),
    strip.placement = "outside",
    legend.position = "top",
    panel.spacing = unit(1.5, "lines"),
    plot.subtitle = element_text(color = "gray40", size = 11),
    
    # --- התיקון הקריטי 3: שוליים חיצוניים ---
    # t=20, b=20 נותן מקום למספרים אם הם בכל זאת בורחים למעלה/למטה
    # l=60 נותן מקום לטקסט בצד שמאל
    plot.margin = margin(t = 20, r = 20, b = 20, l = 60, unit = "pt")
  )

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# ============================================================
# PLOT: The Generalization Gap (With Detection Rate)
# ============================================================

# 1. Prepare Data from 'classified_df'
df_gap_analysis <- classified_df %>%
  # Filter only the specific All-Ponds baselines
  filter(precise_group %in% c("Baseline-2024-All", "Baseline-2025-All")) %>%
  mutate(
    # Determine Training Year based on the group name
    train_year = ifelse(str_detect(precise_group, "2024"), 2024, 2025),
    
    # Define Condition: In-Domain (Good) vs Cross-Domain (Gap)
    condition = ifelse(train_year == test_year, "In-Domain (Same Season)", "Cross-Domain (Generalization Gap)"),
    
    # Create a clean X-axis label
    scenario_label = paste0("Train ", train_year, "\nTest ", test_year)
  ) %>%
  # Select only relevant metrics
  select(scenario_label, condition, MAE_CL, MAE_TL, DR_CL, DR_TL) %>%
  # Pivot to Long Format
  pivot_longer(cols = c(MAE_CL, MAE_TL, DR_CL, DR_TL), 
               names_to = "metric_raw", 
               values_to = "value") %>%
  # Create categorization columns for the Plot
  mutate(
    Body_Part = ifelse(str_detect(metric_raw, "_CL"), "Carapace Length", "Total Length"),
    Metric_Type = ifelse(str_detect(metric_raw, "MAE"), "MAE (Error in mm) ↓", "Detection Rate (%) ↑"),
    # Fix scaling: If Detection Rate is 0-1, convert to 0-100
    value = ifelse(str_detect(metric_raw, "DR") & value <= 1, value * 100, value)
  )

# 2. Generate the Plot
p_gap_final <- ggplot(df_gap_analysis, aes(x = scenario_label, y = value, fill = condition)) +
  geom_col(position = position_dodge(0.8), width = 0.7, color = "black", alpha = 0.85) +
  
  # Add value labels on top of bars
  geom_text(aes(label = round(value, 1)), 
            position = position_dodge(0.8), 
            vjust = -0.5, 
            size = 3.5, 
            fontface = "bold") +
  
  # The Grid Layout: Rows = Metrics, Columns = Body Parts
  facet_grid(Metric_Type ~ Body_Part, scales = "free_y") +
  
  # Manual Colors
  scale_fill_manual(values = c("In-Domain (Same Season)" = "#2a9d8f",    # Green/Teal
                               "Cross-Domain (Generalization Gap)" = "#e76f51")) + # Red/Orange
  
  # Styling
  labs(
    title = "The Generalization Gap: Error vs. Detection Stability",
    subtitle = "Comparing model performance within the same season vs. across seasons",
    y = "Value",
    x = NULL
  ) +
  
  # Add space at top for labels
  scale_y_continuous(expand = expansion(mult = c(0, 0.2))) +
  
  theme_bw(base_size = 12) +
  theme(
    legend.position = "bottom",
    legend.title = element_blank(),
    strip.background = element_rect(fill = "#f0f0f0"),
    strip.text = element_text(face = "bold", size = 11),
    axis.text.x = element_text(face = "bold")
  )

# 3. Print Plot
print(p_gap_final)

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!ֿ
# ============================================================
# PLOT A: Generalization Gap (Final Fix - No Cutting)
# ============================================================

# 1. הכנת הדאטה (ללא כפילויות, כמו שסיכמנו)
df_gap_final <- classified_df %>%
  filter(precise_group %in% c("Baseline-2024-All", "Baseline-2025-All")) %>%
  filter(!str_detect(model, "2024_2025")) %>% 
  mutate(
    train_year = ifelse(str_detect(precise_group, "2024"), 2024, 2025),
    condition = ifelse(train_year == test_year, "In-Domain (Same Season)", "Cross-Domain (Gap)"),
    scenario_label = paste0("Train ", train_year, " -> Test ", test_year)
  ) %>%
  select(scenario_label, condition, MAE_CL, MAE_TL, DR_CL, DR_TL) %>%
  pivot_longer(cols = c(MAE_CL, MAE_TL, DR_CL, DR_TL), 
               names_to = "metric_raw", 
               values_to = "value") %>%
  mutate(
    Body_Part = ifelse(str_detect(metric_raw, "_CL"), "Carapace Length", "Total Length"),
    Metric_Type = ifelse(str_detect(metric_raw, "MAE"), "MAE (Error in mm) ↓", "Detection Rate (%) ↑"),
    value = ifelse(str_detect(metric_raw, "DR") & value <= 1, value * 100, value)
  )

# 2. הציור
p_gap_complete <- ggplot(df_gap_final, aes(x = scenario_label, y = value, fill = condition)) +
  geom_col(position = position_dodge(0.8), width = 0.7, color = "black", alpha = 0.85) +
  
  geom_text(aes(label = round(value, 1)), 
            position = position_dodge(0.8), 
            vjust = -0.5, 
            size = 3.5, 
            fontface = "bold") +
  
  facet_grid(Metric_Type ~ Body_Part, scales = "free_y") +
  
  scale_fill_manual(values = c("In-Domain (Same Season)" = "#2a9d8f", 
                               "Cross-Domain (Gap)" = "#e76f51")) +
  
  labs(
    title = "The Generalization Gap",
    subtitle = "Comparison of Error (MAE) and Detection Rate across seasons",
    y = "Value",
    x = NULL
  ) +
  
  # --- תיקון 1: הגדלת הרווח למעלה ל-35% ---
  scale_y_continuous(expand = expansion(mult = c(0, 0.35))) +
  
  # --- תיקון 2: מניעת חיתוך של טקסט שבורח החוצה ---
  coord_cartesian(clip = "off") +
  
  theme_bw(base_size = 12) +
  theme(
    legend.position = "bottom",
    legend.title = element_blank(),
    strip.background = element_rect(fill = "#f0f0f0"),
    strip.text = element_text(face = "bold"),
    
    # הטיה של הטקסט למטה כדי שלא יהיה צפוף
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = 11, color = "black"),
    
    # הגדלת המרווחים בין הגרפים כדי שהטקסט לא ייגע בכותרות
    panel.spacing = unit(1.5, "lines")
  )

print(p_gap_complete)
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!ֿ

# ============================================================
# FULL ANALYSIS PIPELINE: Prawn Morphometrics Transfer Learning
# Generates Figures 6, 7, 8, and Additional Academic Plots
# ============================================================

# 1. LIBRARIES & SETUP
# ============================================================
library(tidyverse)
library(readxl)
library(janitor)
library(stringr)

# Academic Theme (Clean & Minimalist)
theme_paper <- theme_classic(base_size = 14) +
  theme(
    axis.text = element_text(color = "black"),
    legend.position = "top",
    legend.title = element_blank(),
    strip.background = element_blank(),
    strip.text = element_text(face = "bold", size = 12),
    plot.title = element_text(face = "bold", size = 14)
  )

colors_metric <- c("MAE_CL" = "#264653", "MAE_TL" = "#e76f51") 
labels_metric <- c("MAE_CL" = "Carapace Length", "MAE_TL" = "Total Length")

# 2. DATA LOADING FUNCTIONS
# ============================================================
clean_num <- function(x) {
  x <- as.character(x)
  x <- str_replace_all(x, "[^0-9\\.\\-]", "")
  suppressWarnings(as.numeric(x))
}

infer_year_from_path <- function(path) {
  if (str_detect(path, "test_on_2024")) return(2024L)
  if (str_detect(path, "test_on_2025")) return(2025L)
  NA_integer_
}

process_file <- function(path) {
  year <- infer_year_from_path(path)
  # Suppress messages/warnings from read_excel
  df <- suppressMessages(read_excel(path, sheet = "ALL_MODELS")) %>% clean_names()
  
  # Case A: 2024 Format (Wide)
  if (year == 2024) {
    if (!"model" %in% names(df)) return(tibble())
    df_2024 <- df %>%
      mutate(metric = str_squish(as.character(metric)), 
             model = str_squish(as.character(model)), 
             combined = clean_num(combined)) %>%
      mutate(metric_std = case_when(
        str_detect(metric, "MAE carapace") ~ "MAE_CL",
        str_detect(metric, "MAE total")    ~ "MAE_TL",
        str_detect(metric, "Detection Rate--carapace") ~ "Det_CL",
        str_detect(metric, "Detection Rate--total")    ~ "Det_TL",
        TRUE ~ NA_character_
      )) %>%
      filter(!is.na(metric_std)) %>%
      select(model, metric_std, combined) %>%
      pivot_wider(names_from = metric_std, values_from = combined) %>%
      mutate(test_year = year)
    return(df_2024)
  }
  
  # Case B: 2025 Format (Flat)
  if (year == 2025) {
    col_mae_cl <- names(df)[str_detect(names(df), "mae.*carapace")]
    col_mae_tl <- names(df)[str_detect(names(df), "mae.*total")]
    col_det_cl <- names(df)[str_detect(names(df), "detection.*carapace")]
    col_det_tl <- names(df)[str_detect(names(df), "detection.*total")]
    
    df_2025 <- df %>%
      mutate(model = str_squish(as.character(model))) %>%
      transmute(
        model,
        MAE_CL = if (length(col_mae_cl) > 0) clean_num(.data[[col_mae_cl[1]]]) else NA_real_,
        MAE_TL = if (length(col_mae_tl) > 0) clean_num(.data[[col_mae_tl[1]]]) else NA_real_,
        Det_CL = if (length(col_det_cl) > 0) clean_num(.data[[col_det_cl[1]]]) else NA_real_,
        Det_TL = if (length(col_det_tl) > 0) clean_num(.data[[col_det_tl[1]]]) else NA_real_,
        test_year = year
      )
    return(df_2025)
  }
  return(tibble())
}

# 3. READ FILES
# ============================================================
# Update these paths to your local folders
dir_tf  <- "/Users/taircarmon/Desktop/Uni-Projects/prawn-size-project/scripts/eval/outputs_TF"
dir_out <- "/Users/taircarmon/Desktop/Uni-Projects/prawn-size-project/scripts/eval/outputs"

files <- c(
  list.files(dir_tf,  pattern = "all_models.*\\.xlsx$", recursive = TRUE, full.names = TRUE),
  list.files(dir_out, pattern = "all_models.*\\.xlsx$", recursive = TRUE, full.names = TRUE)
)

raw_data <- map_df(files, process_file)
write_xlsx(raw_data, "/Users/taircarmon/Desktop/Uni-Projects/prawn-size-project/my_exported_data.xlsx")

# 4. CLASSIFICATION & PARSING
# ============================================================
extract_k <- function(s) as.numeric(str_extract(s, "(?<=k)[0-9\\.]+"))
extract_freeze <- function(s) as.numeric(str_extract(s, "(?<=freeze)[0-9]+"))

classified_df <- raw_data %>%
  mutate(
    m = tolower(model),
    k_pct = extract_k(m),
    k_pct = ifelse(k_pct <= 1, k_pct * 100, k_pct), 
    freeze = extract_freeze(m),
    
    direction = case_when(
      str_detect(m, "2024.*to.*2025") ~ "S24->S25",
      str_detect(m, "2025.*to.*2024") ~ "S25->S24",
      TRUE ~ "None"
    ),
    
    # Create a clean Paper_Name for plots
    Paper_Name = case_when(
      str_detect(m, "circulars_weights") ~ "Base-S24-Circ",
      str_detect(m, "square_weights")    ~ "Base-S24-Sq",
      str_detect(m, "all-ponds")         ~ "Base-S24",
      str_detect(m, "2025_all") & !str_detect(m, "to") ~ "Base-S25",
      str_detect(m, "2024_2025")         ~ "Joint-S24+S25",
      str_detect(m, "circular_to_square") ~ "TL-Circ->Sq",
      str_detect(m, "b2") ~ "TL-Tune",
      str_detect(m, "b1") ~ "TL-Feat",
      TRUE ~ "Other"
    ),
    
    model_type = case_when(
      str_detect(Paper_Name, "Base") ~ Paper_Name,
      str_detect(m, "b1") ~ "TL-Feat",
      str_detect(m, "b2") ~ "TL-Tune",
      TRUE ~ "Other"
    )
  ) %>%
  # NEW: Calculate combined metrics including Detection Rate
  mutate(
    Avg_Error = (MAE_CL + MAE_TL) / 2, 
    Avg_Det = (Det_CL + Det_TL) / 2,
    Fitness_Score = Avg_Error - (0.01 * Avg_Det)
  )

# Define "TL-Optim" as the single best TL-Tune model per direction/year 
tl_optim_candidates <- classified_df %>%
  filter(model_type == "TL-Tune") %>%
  group_by(test_year, direction) %>%
  slice_min(Fitness_Score, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  mutate(model_type = "TL-Optim") 

final_df <- bind_rows(classified_df, tl_optim_candidates)


# ============================================================
# 9. ADDITIONAL ACADEMIC PLOTS (SIMPLIFIED & CLEAN)
# ============================================================

# ------------------------------------------------------------
# PLOT A: THE GENERALIZATION GAP
# Style: Grouped Bar Chart
# ------------------------------------------------------------
df_gap <- final_df %>%
  filter(Paper_Name %in% c("Base-S24", "Base-S25")) %>%
  mutate(
    Test_Set = paste("Test on", test_year),
    Model_Label = ifelse(Paper_Name == "Base-S24", "Model: Train 2024", "Model: Train 2025")
  ) %>%
  pivot_longer(cols = c(MAE_CL, MAE_TL), names_to = "Metric", values_to = "Error") %>%
  mutate(Metric_Label = labels_metric[Metric])

p_gap <- ggplot(df_gap, aes(x = Test_Set, y = Error, fill = Model_Label)) +
  geom_col(position = position_dodge(0.8), width = 0.6, color="black", alpha=0.8) +
  geom_text(aes(label = round(Error, 1)), position = position_dodge(0.8), vjust = -0.5, size=3.5, fontface="bold") +
  scale_fill_manual(values = c("Model: Train 2024" = "#264653", "Model: Train 2025" = "#e76f51")) +
  facet_wrap(~Metric_Label, scales = "free_y") +
  labs(title = "Cross-Season Generalization Gap", y = "Mean Absolute Error (mm)", x = "") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
  theme_paper

print(p_gap)


# ------------------------------------------------------------
# PLOT B: DATA EFFICIENCY (TREND LINE)
# Style: Line Plot with Points (Cleaner for trends)
# ------------------------------------------------------------
# Get Baseline Reference (Horizontal Line)
ref_vals <- final_df %>% 
  filter(Paper_Name == "Base-S25", test_year == 2025) %>%
  select(MAE_CL, MAE_TL) %>%
  pivot_longer(everything(), names_to = "Metric", values_to = "Baseline") %>%
  mutate(Metric_Label = labels_metric[Metric])

df_eff <- final_df %>%
  filter(direction == "S24->S25", model_type == "TL-Tune") %>%
  group_by(k_pct) %>%
  slice_min(Fitness_Score, n = 1) %>%
  ungroup() %>%
  pivot_longer(cols = c(MAE_CL, MAE_TL), names_to = "Metric", values_to = "Error") %>%
  mutate(Metric_Label = labels_metric[Metric])

p_eff <- ggplot(df_eff, aes(x = k_pct, y = Error, color = Metric_Label)) +
  # Add Reference Line (Baseline)
  geom_hline(data = ref_vals, aes(yintercept = Baseline), linetype = "dashed", color = "gray50") +
  # Add Trend Line
  geom_line(linewidth = 1.2) +
  geom_point(size = 4) +
  geom_text(aes(label = round(Error, 1)), vjust = -1.2, fontface="bold", show.legend=FALSE) +
  facet_wrap(~Metric_Label, scales = "free_y") +
  scale_x_continuous(breaks = c(25, 50, 100), labels = c("25%", "50%", "100%")) +
  scale_color_manual(values = c("#2a9d8f", "#e9c46a")) +
  labs(
    title = "Data Efficiency: Transfer Learning vs. Data Budget",
    subtitle = "Dashed line represents model trained on 100% of target data (Baseline)",
    y = "Mean Absolute Error (mm)",
    x = "Percentage of Target Data Used"
  ) +
  theme_paper +
  theme(legend.position = "none") # Colors explained by facets

print(p_eff)


# ------------------------------------------------------------
# PLOT C: FREEZE DEPTH (TREND LINES BY DATA %)
# Style: Line Plot with Points, Separate Line for each k%
# ------------------------------------------------------------
library(tidyverse)
library(stringr)

# ============================================================
# 1. פונקציית ניקוי ופרסור
# ============================================================
parse_model_info <- function(df) {
  df %>%
    mutate(
      Architecture = case_when(
        str_detect(model, "B1") ~ "B1",
        str_detect(model, "B2") ~ "B2",
        TRUE ~ "Other"
      ),
      Direction = case_when(
        str_detect(model, "2024to2025") ~ "Train 24 -> Target 25",
        str_detect(model, "2025to2024") ~ "Train 25 -> Target 24",
        TRUE ~ "Other"
      ),
      Weight_Type = case_when(
        str_detect(model, "best") ~ "Best",
        str_detect(model, "last") ~ "Last",
        !str_detect(model, "\\.pt") & !str_detect(model, "weights") ~ "Clean_Name", 
        TRUE ~ "Other"
      ),
      # חילוץ Seed - אם לא מוצא כלום, נניח שזה S1 (ברירת מחדל ל-B1)
      Seed = str_extract(model, "S[0-9]+"),
      Seed = ifelse(is.na(Seed), "S1", Seed), 
      
      k_pct = as.numeric(str_extract(model, "(?<=k)[0-9]+")),
      raw_freeze = as.numeric(str_extract(model, "(?<=freeze)[0-9]+")),
      freeze = case_when(
        raw_freeze == 82 ~ 8,
        raw_freeze == 42 ~ 4,
        raw_freeze == 02 ~ 0,
        TRUE ~ raw_freeze
      )
    )
}

# ============================================================
# 2. הסינון החכם (The Smart Hybrid Filter)
# ============================================================
final_analysis_df <- raw_data %>%
  parse_model_info() %>%
  
  # --- סינון סוג משקולות (מעיף Last) ---
  filter(Weight_Type %in% c("Best", "Clean_Name")) %>%
  
  # --- סינון ארכיטקטורה ו-Seed (החלק החשוב!) ---
  filter(
    (Architecture == "B2" & Seed == "S2") |  # עבור B2, תהיה קשוח וקח רק S2
      (Architecture == "B1")                   # עבור B1, קח מה שיש (S1)
  ) %>%
  
  # --- סינון שנת טסט ---
  filter(
    (Direction == "Train 24 -> Target 25" & test_year == 2025) |
      (Direction == "Train 25 -> Target 24" & test_year == 2024)
  ) %>%
  
  # --- סינון פרמטרים ---
  filter(!is.na(freeze), !is.na(k_pct)) %>%
  
  mutate(
    Scenario_Label = ifelse(Direction == "Train 24 -> Target 25", 
                            "Target: 2025 (Transfer from 24)", 
                            "Target: 2024 (Transfer from 25)")
  )

# ============================================================
# 3. בדיקה - מה נשאר?
# ============================================================
cat("\n--- בדיקת כמויות לפי ארכיטקטורה ו-Seed ---\n")
print(table(final_analysis_df$Architecture, final_analysis_df$Seed))

# שמירת התוצאה
df_truth <- final_analysis_df
# ============================================================
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!new
# ============================================================
library(tidyverse)
library(stringr)

# ==============================================================================
# 1. RAW DATA: הרשימה המלאה (248 מודלים)
# ==============================================================================
raw_model_list <- c(
  "B1_TL_2024to2025_k100pct_freeze02",
  "B1_TL_2024to2025_k100pct_freeze42",
  "B1_TL_2024to2025_k100pct_freeze82",
  "B1_TL_2024to2025_k100pct_freezeheadsonly2",
  "B1_TL_2024to2025_k100pct_freezeheadsonly3",
  "B1_TL_2024to2025_k25pct_freeze02",
  "B1_TL_2024to2025_k25pct_freeze42",
  "B1_TL_2024to2025_k25pct_freeze82",
  "B1_TL_2024to2025_k25pct_freezeheadsonly2",
  "B1_TL_2024to2025_k50pct_freeze02",
  "B1_TL_2024to2025_k50pct_freeze42",
  "B1_TL_2024to2025_k50pct_freeze82",
  "B1_TL_2024to2025_k50pct_freezeheadsonly2",
  "B1_TL_2025to2024_k100pct_freeze02",
  "B1_TL_2025to2024_k100pct_freeze42",
  "B1_TL_2025to2024_k100pct_freeze82",
  "B1_TL_2025to2024_k100pct_freezeheadsonly2",
  "B1_TL_2025to2024_k25pct_freeze02",
  "B1_TL_2025to2024_k25pct_freeze42",
  "B1_TL_2025to2024_k25pct_freeze82",
  "B1_TL_2025to2024_k25pct_freezeheadsonly2",
  "B1_TL_2025to2024_k50pct_freeze02",
  "B1_TL_2025to2024_k50pct_freeze42",
  "B1_TL_2025to2024_k50pct_freeze82",
  "B1_TL_2025to2024_k50pct_freezeheadsonly2",
  "from_2025_to_2024_B1_TL_2025to2024_k25pct_freeze42_weights_last.pt",
  "from_2025_to_2024_B1_TL_2025to2024_k25pct_freeze42_weights_best.pt",
  "from_2025_to_2024_B1_TL_2025to2024_k50pct_freeze42_weights_last.pt",
  "from_2025_to_2024_B1_TL_2025to2024_k50pct_freeze42_weights_best.pt",
  "from_2025_to_2024_B1_TL_2025to2024_k100pct_freeze02_weights_last.pt",
  "from_2025_to_2024_B1_TL_2025to2024_k100pct_freeze02_weights_best.pt",
  "from_2025_to_2024_B1_TL_2025to2024_k100pct_freeze42_weights_last.pt",
  "from_2025_to_2024_B1_TL_2025to2024_k100pct_freeze42_weights_best.pt",
  "from_2025_to_2024_B1_TL_2025to2024_k50pct_freeze02_weights_last.pt",
  "from_2025_to_2024_B1_TL_2025to2024_k50pct_freeze02_weights_best.pt",
  "from_2025_to_2024_B1_TL_2025to2024_k25pct_freezeheadsonly2_weights_last.pt",
  "from_2025_to_2024_B1_TL_2025to2024_k25pct_freezeheadsonly2_weights_best.pt",
  "from_2025_to_2024_B1_TL_2025to2024_k25pct_freeze02_weights_last.pt",
  "from_2025_to_2024_B1_TL_2025to2024_k25pct_freeze02_weights_best.pt",
  "from_2025_to_2024_B1_TL_2025to2024_k100pct_freezeheadsonly2_weights_last.pt",
  "from_2025_to_2024_B1_TL_2025to2024_k100pct_freezeheadsonly2_weights_best.pt",
  "from_2025_to_2024_B1_TL_2025to2024_k25pct_freeze82_weights_last.pt",
  "from_2025_to_2024_B1_TL_2025to2024_k25pct_freeze82_weights_best.pt",
  "from_2025_to_2024_B1_TL_2025to2024_k50pct_freeze82_weights_last.pt",
  "from_2025_to_2024_B1_TL_2025to2024_k50pct_freeze82_weights_best.pt",
  "from_2025_to_2024_B1_TL_2025to2024_k100pct_freeze82_weights_last.pt",
  "from_2025_to_2024_B1_TL_2025to2024_k100pct_freeze82_weights_best.pt",
  "from_2025_to_2024_B1_TL_2025to2024_k50pct_freezeheadsonly2_weights_last.pt",
  "from_2025_to_2024_B1_TL_2025to2024_k50pct_freezeheadsonly2_weights_best.pt",
  "from_2024_to_2025_B1_TL_2024to2025_k50pct_freezeheadsonly2_weights_last.pt",
  "from_2024_to_2025_B1_TL_2024to2025_k50pct_freezeheadsonly2_weights_best.pt",
  "from_2024_to_2025_B1_TL_2024to2025_k100pct_freeze02_weights_last.pt",
  "from_2024_to_2025_B1_TL_2024to2025_k100pct_freeze02_weights_best.pt",
  "from_2024_to_2025_B1_TL_2024to2025_k50pct_freeze82_weights_last.pt",
  "from_2024_to_2025_B1_TL_2024to2025_k50pct_freeze82_weights_best.pt",
  "from_2024_to_2025_B1_TL_2024to2025_k100pct_freezeheadsonly3_weights_last.pt",
  "from_2024_to_2025_B1_TL_2024to2025_k100pct_freezeheadsonly3_weights_best.pt",
  "from_2024_to_2025_B1_TL_2024to2025_k100pct_freezeheadsonly2_weights_last.pt",
  "from_2024_to_2025_B1_TL_2024to2025_k100pct_freezeheadsonly2_weights_best.pt",
  "from_2024_to_2025_B1_TL_2024to2025_k100pct_freeze42_weights_last.pt",
  "from_2024_to_2025_B1_TL_2024to2025_k100pct_freeze42_weights_best.pt",
  "from_2024_to_2025_B1_TL_2024to2025_k25pct_freeze82_weights_last.pt",
  "from_2024_to_2025_B1_TL_2024to2025_k25pct_freeze82_weights_best.pt",
  "from_2024_to_2025_B1_TL_2024to2025_k25pct_freezeheadsonly2_weights_last.pt",
  "from_2024_to_2025_B1_TL_2024to2025_k25pct_freezeheadsonly2_weights_best.pt",
  "from_2024_to_2025_B1_TL_2024to2025_k25pct_freeze02_weights_last.pt",
  "from_2024_to_2025_B1_TL_2024to2025_k25pct_freeze02_weights_best.pt",
  "from_2024_to_2025_B1_TL_2024to2025_k50pct_freeze02_weights_last.pt",
  "from_2024_to_2025_B1_TL_2024to2025_k50pct_freeze02_weights_best.pt",
  "from_2024_to_2025_B1_TL_2024to2025_k50pct_freeze42_weights_last.pt",
  "from_2024_to_2025_B1_TL_2024to2025_k50pct_freeze42_weights_best.pt",
  "from_2024_to_2025_B1_TL_2024to2025_k25pct_freeze42_weights_last.pt",
  "from_2024_to_2025_B1_TL_2024to2025_k25pct_freeze42_weights_best.pt",
  "from_2024_to_2025_B1_TL_2024to2025_k100pct_freeze82_weights_last.pt",
  "from_2024_to_2025_B1_TL_2024to2025_k100pct_freeze82_weights_best.pt",
  "B2_TL_2024to2025_k100pct_freeze10_S13",
  "B2_TL_2024to2025_k100pct_freeze4_S1",
  "B2_TL_2024to2025_k100pct_freeze4_S2",
  "B2_TL_2024to2025_k100pct_freeze8_S1",
  "B2_TL_2024to2025_k100pct_freeze8_S2",
  "B2_TL_2024to2025_k25pct_freeze10_S1",
  "B2_TL_2024to2025_k25pct_freeze10_S2",
  "B2_TL_2024to2025_k25pct_freeze4_S1",
  "B2_TL_2024to2025_k25pct_freeze4_S2",
  "B2_TL_2024to2025_k25pct_freeze8_S1",
  "B2_TL_2024to2025_k25pct_freeze8_S2",
  "B2_TL_2024to2025_k50pct_freeze10_S1",
  "B2_TL_2024to2025_k50pct_freeze10_S2",
  "B2_TL_2024to2025_k50pct_freeze4_S1",
  "B2_TL_2024to2025_k50pct_freeze4_S2",
  "B2_TL_2024to2025_k50pct_freeze8_S1",
  "B2_TL_2024to2025_k50pct_freeze8_S2",
  "B2_TL_2025to2024_k100pct_freeze10_S1",
  "B2_TL_2025to2024_k100pct_freeze10_S2",
  "B2_TL_2025to2024_k100pct_freeze4_S1",
  "B2_TL_2025to2024_k100pct_freeze4_S2",
  "B2_TL_2025to2024_k100pct_freeze8_S1",
  "B2_TL_2025to2024_k100pct_freeze8_S2",
  "B2_TL_2025to2024_k25pct_freeze10_S1",
  "B2_TL_2025to2024_k25pct_freeze10_S2",
  "B2_TL_2025to2024_k25pct_freeze4_S1",
  "B2_TL_2025to2024_k25pct_freeze4_S2",
  "B2_TL_2025to2024_k25pct_freeze8_S1",
  "B2_TL_2025to2024_k25pct_freeze8_S2",
  "B2_TL_2025to2024_k50pct_freeze10_S1",
  "B2_TL_2025to2024_k50pct_freeze10_S2",
  "B2_TL_2025to2024_k50pct_freeze4_S1",
  "B2_TL_2025to2024_k50pct_freeze4_S2",
  "B2_TL_2025to2024_k50pct_freeze8_S1",
  "B2_TL_2025to2024_k50pct_freeze8_S2",
  "from_2024to2025_B2_TL_2024to2025_k100pct_freeze4_S1",
  "from_2024to2025_B2_TL_2024to2025_k100pct_freeze4_S2",
  "from_2024to2025_B2_TL_2024to2025_k100pct_freeze8_S1",
  "from_2024to2025_B2_TL_2024to2025_k100pct_freeze8_S2",
  "from_2024to2025_B2_TL_2024to2025_k25pct_freeze10_S1",
  "from_2024to2025_B2_TL_2024to2025_k25pct_freeze10_S2",
  "from_2024to2025_B2_TL_2024to2025_k25pct_freeze4_S1",
  "from_2024to2025_B2_TL_2024to2025_k25pct_freeze4_S2",
  "from_2024to2025_B2_TL_2024to2025_k25pct_freeze8_S1",
  "from_2024to2025_B2_TL_2024to2025_k25pct_freeze8_S2",
  "from_2024to2025_B2_TL_2024to2025_k50pct_freeze10_S1",
  "from_2024to2025_B2_TL_2024to2025_k50pct_freeze10_S2",
  "from_2024to2025_B2_TL_2024to2025_k50pct_freeze4_S1",
  "from_2024to2025_B2_TL_2024to2025_k50pct_freeze4_S2",
  "from_2024to2025_B2_TL_2024to2025_k50pct_freeze8_S1",
  "from_2024to2025_B2_TL_2024to2025_k50pct_freeze8_S2",
  "from_2025to2024_B2_TL_2025to2024_k100pct_freeze10_S1",
  "from_2025to2024_B2_TL_2025to2024_k100pct_freeze10_S2",
  "from_2025to2024_B2_TL_2025to2024_k100pct_freeze4_S1",
  "from_2025to2024_B2_TL_2025to2024_k100pct_freeze4_S2",
  "from_2025to2024_B2_TL_2025to2024_k100pct_freeze8_S1",
  "from_2025to2024_B2_TL_2025to2024_k100pct_freeze8_S2",
  "from_2025to2024_B2_TL_2025to2024_k25pct_freeze10_S1",
  "from_2025to2024_B2_TL_2025to2024_k25pct_freeze10_S2",
  "from_2025to2024_B2_TL_2025to2024_k25pct_freeze4_S1",
  "from_2025to2024_B2_TL_2025to2024_k25pct_freeze4_S2",
  "from_2025to2024_B2_TL_2025to2024_k25pct_freeze8_S1",
  "from_2025to2024_B2_TL_2025to2024_k25pct_freeze8_S2",
  "from_2025to2024_B2_TL_2025to2024_k50pct_freeze10_S1",
  "from_2025to2024_B2_TL_2025to2024_k50pct_freeze10_S2",
  "from_2025to2024_B2_TL_2025to2024_k50pct_freeze4_S1",
  "from_2025to2024_B2_TL_2025to2024_k50pct_freeze4_S2",
  "from_2025to2024_B2_TL_2025to2024_k50pct_freeze8_S1",
  "from_2025to2024_B2_TL_2025to2024_k50pct_freeze8_S2",
  "from_2024to2025_B2_TL_2024to2025_k25pct_freeze4_S2_best.pt",
  "from_2024to2025_B2_TL_2024to2025_k25pct_freeze10_S1_best.pt",
  "from_2024to2025_B2_TL_2024to2025_k50pct_freeze8_S1_best.pt",
  "from_2024to2025_B2_TL_2024to2025_k50pct_freeze10_S1_best.pt",
  "from_2024to2025_B2_TL_2024to2025_k100pct_freeze8_S1_best.pt",
  "from_2024to2025_B2_TL_2024to2025_k25pct_freeze4_S1_best.pt",
  "from_2024to2025_B2_TL_2024to2025_k100pct_freeze8_S2_best.pt",
  "from_2024to2025_B2_TL_2024to2025_k50pct_freeze10_S2_best.pt",
  "from_2024to2025_B2_TL_2024to2025_k50pct_freeze8_S2_best.pt",
  "from_2024to2025_B2_TL_2024to2025_k25pct_freeze10_S2_best.pt",
  "from_2024to2025_B2_TL_2024to2025_k100pct_freeze4_S1_best.pt",
  "from_2024to2025_B2_TL_2024to2025_k50pct_freeze4_S1_best.pt",
  "from_2024to2025_B2_TL_2024to2025_k25pct_freeze8_S2_best.pt",
  "from_2024to2025_B2_TL_2024to2025_k50pct_freeze4_S2_best.pt",
  "from_2024to2025_B2_TL_2024to2025_k100pct_freeze4_S2_best.pt",
  "from_2024to2025_B2_TL_2024to2025_k25pct_freeze8_S1_best.pt",
  "from_2025to2024_B2_TL_2025to2024_k100pct_freeze4_S2_best.pt",
  "from_2025to2024_B2_TL_2025to2024_k50pct_freeze8_S1_best.pt",
  "from_2025to2024_B2_TL_2025to2024_k100pct_freeze10_S2_best.pt",
  "from_2025to2024_B2_TL_2025to2024_k25pct_freeze4_S2_best.pt",
  "from_2025to2024_B2_TL_2025to2024_k100pct_freeze10_S1_best.pt",
  "from_2025to2024_B2_TL_2025to2024_k100pct_freeze4_S1_best.pt",
  "from_2025to2024_B2_TL_2025to2024_k50pct_freeze8_S2_best.pt",
  "from_2025to2024_B2_TL_2025to2024_k25pct_freeze4_S1_best.pt",
  "from_2025to2024_B2_TL_2025to2024_k25pct_freeze8_S2_best.pt",
  "from_2025to2024_B2_TL_2025to2024_k25pct_freeze10_S2_best.pt",
  "from_2025to2024_B2_TL_2025to2024_k50pct_freeze10_S2_best.pt",
  "from_2025to2024_B2_TL_2025to2024_k100pct_freeze8_S2_best.pt",
  "from_2025to2024_B2_TL_2025to2024_k50pct_freeze4_S1_best.pt",
  "from_2025to2024_B2_TL_2025to2024_k25pct_freeze8_S1_best.pt",
  "from_2025to2024_B2_TL_2025to2024_k100pct_freeze8_S1_best.pt",
  "from_2025to2024_B2_TL_2025to2024_k50pct_freeze4_S2_best.pt",
  "from_2025to2024_B2_TL_2025to2024_k50pct_freeze10_S1_best.pt",
  "from_2025to2024_B2_TL_2025to2024_k25pct_freeze10_S1_best.pt",
  "from_2024to2025_B2_TL_2024to2025_k25pct_freeze10_S1_weights_last.pt",
  "from_2024to2025_B2_TL_2024to2025_k25pct_freeze10_S1_weights_best.pt",
  "from_2024to2025_B2_TL_2024to2025_k50pct_freeze4_S1_weights_last.pt",
  "from_2024to2025_B2_TL_2024to2025_k50pct_freeze4_S1_weights_best.pt",
  "from_2024to2025_B2_TL_2024to2025_k50pct_freeze8_S2_weights_last.pt",
  "from_2024to2025_B2_TL_2024to2025_k50pct_freeze8_S2_weights_best.pt",
  "from_2024to2025_B2_TL_2024to2025_k25pct_freeze8_S1_weights_last.pt",
  "from_2024to2025_B2_TL_2024to2025_k25pct_freeze8_S1_weights_best.pt",
  "from_2024to2025_B2_TL_2024to2025_k25pct_freeze4_S2_weights_last.pt",
  "from_2024to2025_B2_TL_2024to2025_k25pct_freeze4_S2_weights_best.pt",
  "from_2024to2025_B2_TL_2024to2025_k100pct_freeze8_S2_weights_last.pt",
  "from_2024to2025_B2_TL_2024to2025_k100pct_freeze8_S2_weights_best.pt",
  "from_2024to2025_B2_TL_2024to2025_k100pct_freeze4_S1_weights_last.pt",
  "from_2024to2025_B2_TL_2024to2025_k100pct_freeze4_S1_weights_best.pt",
  "from_2024to2025_B2_TL_2024to2025_k100pct_freeze10_S13_weights_last.pt",
  "from_2024to2025_B2_TL_2024to2025_k100pct_freeze10_S13_weights_best.pt",
  "from_2024to2025_B2_TL_2024to2025_k50pct_freeze10_S1_weights_last.pt",
  "from_2024to2025_B2_TL_2024to2025_k50pct_freeze10_S1_weights_best.pt",
  "from_2024to2025_B2_TL_2024to2025_k25pct_freeze10_S2_weights_last.pt",
  "from_2024to2025_B2_TL_2024to2025_k25pct_freeze10_S2_weights_best.pt",
  "from_2024to2025_B2_TL_2024to2025_k50pct_freeze8_S1_weights_last.pt",
  "from_2024to2025_B2_TL_2024to2025_k50pct_freeze8_S1_weights_best.pt",
  "from_2024to2025_B2_TL_2024to2025_k50pct_freeze4_S2_weights_last.pt",
  "from_2024to2025_B2_TL_2024to2025_k50pct_freeze4_S2_weights_best.pt",
  "from_2024to2025_B2_TL_2024to2025_k25pct_freeze4_S1_weights_last.pt",
  "from_2024to2025_B2_TL_2024to2025_k25pct_freeze4_S1_weights_best.pt",
  "from_2024to2025_B2_TL_2024to2025_k25pct_freeze8_S2_weights_last.pt",
  "from_2024to2025_B2_TL_2024to2025_k25pct_freeze8_S2_weights_best.pt",
  "from_2024to2025_B2_TL_2024to2025_k100pct_freeze4_S2_weights_last.pt",
  "from_2024to2025_B2_TL_2024to2025_k100pct_freeze4_S2_weights_best.pt",
  "from_2024to2025_B2_TL_2024to2025_k100pct_freeze8_S1_weights_last.pt",
  "from_2024to2025_B2_TL_2024to2025_k100pct_freeze8_S1_weights_best.pt",
  "from_2024to2025_B2_TL_2024to2025_k50pct_freeze10_S2_weights_last.pt",
  "from_2024to2025_B2_TL_2024to2025_k50pct_freeze10_S2_weights_best.pt",
  "from_2025to2024_B2_TL_2025to2024_k50pct_freeze4_S2_weights_last.pt",
  "from_2025to2024_B2_TL_2025to2024_k50pct_freeze4_S2_weights_best.pt",
  "from_2025to2024_B2_TL_2025to2024_k25pct_freeze10_S2_weights_last.pt",
  "from_2025to2024_B2_TL_2025to2024_k25pct_freeze10_S2_weights_best.pt",
  "from_2025to2024_B2_TL_2025to2024_k50pct_freeze8_S1_weights_last.pt",
  "from_2025to2024_B2_TL_2025to2024_k50pct_freeze8_S1_weights_best.pt",
  "from_2025to2024_B2_TL_2025to2024_k25pct_freeze8_S2_weights_last.pt",
  "from_2025to2024_B2_TL_2025to2024_k25pct_freeze8_S2_weights_best.pt",
  "from_2025to2024_B2_TL_2025to2024_k25pct_freeze4_S1_weights_last.pt",
  "from_2025to2024_B2_TL_2025to2024_k25pct_freeze4_S1_weights_best.pt",
  "from_2025to2024_B2_TL_2025to2024_k100pct_freeze4_S2_weights_last.pt",
  "from_2025to2024_B2_TL_2025to2024_k100pct_freeze4_S2_weights_best.pt",
  "from_2025to2024_B2_TL_2025to2024_k100pct_freeze10_S1_weights_last.pt",
  "from_2025to2024_B2_TL_2025to2024_k100pct_freeze10_S1_weights_best.pt",
  "from_2025to2024_B2_TL_2025to2024_k100pct_freeze8_S1_weights_last.pt",
  "from_2025to2024_B2_TL_2025to2024_k100pct_freeze8_S1_weights_best.pt",
  "from_2025to2024_B2_TL_2025to2024_k50pct_freeze10_S2_weights_last.pt",
  "from_2025to2024_B2_TL_2025to2024_k50pct_freeze10_S2_weights_best.pt",
  "from_2025to2024_B2_TL_2025to2024_k25pct_freeze10_S1_weights_last.pt",
  "from_2025to2024_B2_TL_2025to2024_k25pct_freeze10_S1_weights_best.pt",
  "from_2025to2024_B2_TL_2025to2024_k50pct_freeze8_S2_weights_last.pt",
  "from_2025to2024_B2_TL_2025to2024_k50pct_freeze8_S2_weights_best.pt",
  "from_2025to2024_B2_TL_2025to2024_k50pct_freeze4_S1_weights_last.pt",
  "from_2025to2024_B2_TL_2025to2024_k50pct_freeze4_S1_weights_best.pt",
  "from_2025to2024_B2_TL_2025to2024_k25pct_freeze4_S2_weights_last.pt",
  "from_2025to2024_B2_TL_2025to2024_k25pct_freeze4_S2_weights_best.pt",
  "from_2025to2024_B2_TL_2025to2024_k25pct_freeze8_S1_weights_last.pt",
  "from_2025to2024_B2_TL_2025to2024_k25pct_freeze8_S1_weights_best.pt",
  "from_2025to2024_B2_TL_2025to2024_k100pct_freeze8_S2_weights_last.pt",
  "from_2025to2024_B2_TL_2025to2024_k100pct_freeze8_S2_weights_best.pt",
  "from_2025to2024_B2_TL_2025to2024_k100pct_freeze10_S2_weights_last.pt",
  "from_2025to2024_B2_TL_2025to2024_k100pct_freeze10_S2_weights_best.pt",
  "from_2025to2024_B2_TL_2025to2024_k100pct_freeze4_S1_weights_last.pt",
  "from_2025to2024_B2_TL_2025to2024_k100pct_freeze4_S1_weights_best.pt",
  "from_2025to2024_B2_TL_2025to2024_k50pct_freeze10_S1_weights_last.pt",
  "from_2025to2024_B2_TL_2025to2024_k50pct_freeze10_S1_weights_best.pt"
)

# ==============================================================================
# 2. פונקציית ייצור ה-TRUTH TABLE
# ==============================================================================
generate_truth_table <- function(model_list) {
  
  # המרת רשימה לטבלה
  df_truth <- tibble(original = model_list)
  
  df_truth <- df_truth %>%
    mutate(
      # --- שלב א: ניקוי הסטרינג ---
      clean_id = original %>%
        str_remove("^from_\\d+(_)?to(_)?_\\d+_") %>% 
        str_remove("^from_\\d+to\\d+_") %>%
        str_remove("(_weights)?_(best|last)\\.pt$"),
      
      # --- שלב ב: חילוץ פרמטרים ---
      Batch = str_extract(clean_id, "^B\\d"), 
      
      Direction = str_extract(clean_id, "\\d+to\\d+"),
      Target_Year = case_when(
        Direction == "2024to2025" ~ 2025,
        Direction == "2025to2024" ~ 2024,
        TRUE ~ NA_real_
      ),
      
      k_pct = as.numeric(str_extract(clean_id, "(?<=k)\\d+")),
      
      Freeze_Raw_String = str_extract(clean_id, "(?<=freeze)[a-zA-Z0-9]+"),
      
      # חילוץ Seed (ברירת מחדל S1)
      Seed_Raw = str_extract(clean_id, "S\\d+$"),
      Seed = ifelse(is.na(Seed_Raw) | Seed_Raw == "", "S1", Seed_Raw)
    ) %>%
    
    # --- שלב ג: תיקון המספרים (42 -> 4) ---
    mutate(
      Freeze_Num = as.numeric(str_extract(Freeze_Raw_String, "\\d+")),
      
      Freeze_Final = case_when(
        str_detect(Freeze_Raw_String, "headsonly") ~ 0,
        Freeze_Num == 82 ~ 8,
        Freeze_Num == 42 ~ 4,
        Freeze_Num == 02 ~ 0,
        TRUE ~ Freeze_Num
      )
    ) %>%
    
    # בחירת עמודות סופית
    select(
      Original_String = original,
      Batch,
      Target_Year,
      k_pct,
      Freeze_Final,
      Seed,
      Clean_ID = clean_id
    ) %>%
    
    # מיון
    arrange(Target_Year, Batch, k_pct, Freeze_Final, Seed)
  
  return(df_truth)
}

# ==============================================================================
# 3. הרצה
# ==============================================================================
truth_table <- generate_truth_table(raw_model_list)

# ==============================================================================
# 4. הדפסת הסיכום (Validation) - הוכחה שאין חוסרים
# ==============================================================================
cat("\n=================================================\n")
cat(" COVERAGE SUMMARY (Target Year x Freeze)\n")
cat("=================================================\n")

summary_view <- truth_table %>%
  filter(!is.na(Target_Year)) %>% # רק מודלים תקינים
  group_by(Target_Year, Batch, Freeze_Final) %>%
  summarise(
    # מראה אילו k קיימים (מצפים לראות 25, 50, 100 בכל שורה)
    k_Values_Found = paste(sort(unique(k_pct)), collapse=", "),
    
    # מראה אילו Seeds קיימים (לפחות אחד מבין S1, S2, S13)
    Seeds_Found = paste(sort(unique(Seed)), collapse=", "),
    
    Count = n(),
    .groups = "drop"
  )

print(as.data.frame(summary_view))


# ==============================================================================
#grao
# ==============================================================================
library(tidyverse)
library(ggplot2)

# ============================================================
# 1. הכנת הנתונים (ישירות מ-raw_data שיש לך בזיכרון)
# ============================================================

df_gap_final <- raw_data %>%
  mutate(m = tolower(model)) %>%
  
  # --- א. זיהוי שנת האימון (Train Year) ---
  mutate(
    Train_Year = case_when(
      # מודלים של העברה (Transfer Learning)
      str_detect(m, "2024to2025") ~ 2024,
      str_detect(m, "2025to2024") ~ 2025,
      str_detect(m, "from_2024") ~ 2024,
      str_detect(m, "from_2025") ~ 2025,
      
      # מודלים של בייסליין (אימון רגיל)
      str_detect(m, "all-ponds") ~ 2024,
      str_detect(m, "2025_all") & !str_detect(m, "to") ~ 2025,
      
      TRUE ~ NA_real_
    )
  ) %>%
  filter(!is.na(Train_Year)) %>%
  
  # --- ב. הגדרת התנאים לגרף ---
  mutate(
    Condition = ifelse(Train_Year == test_year, "Native (Same Year)", "External (Cross-Year)"),
    Target_Label = paste0("Test Target: ", test_year)
  ) %>%
  
  # --- ג. סינון: לוקחים את המודלים "הכי חזקים" להשוואה ---
  # (בייסליינים או מודלים שהשתמשו ב-100% מהדאטה כדי לייצג את השנה)
  filter(
    str_detect(m, "all-ponds") | 
      str_detect(m, "2025_all") |
      str_detect(m, "k100pct")  # לוקח את שורות ה-B1 שהראית לי
  ) %>%
  
  # --- ד. מניעת כפילויות ---
  # לוקחים את המודל עם השגיאה הכי נמוכה לכל קומבינציה
  group_by(Target_Label, Train_Year) %>%
  slice_min(MAE_TL, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  
  # --- ה. סידור לפורמט ארוך לציור ---
  # משתמשים ישירות בשמות העמודות הנכונים שלך!
  select(Target_Label, Train_Year, Condition, MAE_CL, MAE_TL, DR_CL, DR_TL) %>%
  pivot_longer(cols = c(MAE_CL, MAE_TL, DR_CL, DR_TL), 
               names_to = "metric_raw", 
               values_to = "value") %>%
  mutate(
    Body_Part = ifelse(str_detect(metric_raw, "_CL"), "Carapace", "Total"),
    Metric_Type = ifelse(str_detect(metric_raw, "MAE"), "Error (MAE) ↓", "Detection Rate (%) ↑"),
    
    # בדיקה אם צריך להמיר לאחוזים (אם המספר הוא 0.9 -> 90, אם הוא 90 -> נשאר 90)
    value = ifelse(str_detect(metric_raw, "DR") & value <= 1, value * 100, value)
  )

# ============================================================
# 2. הציור
# ============================================================

p_gap <- ggplot(df_gap_final, aes(x = as.factor(Train_Year), y = value, fill = Condition)) +
  
  # עמודות
  geom_col(position = position_dodge(0.8), width = 0.7, color = "black", alpha = 0.85) +
  
  # טקסט מעל העמודות
  geom_text(aes(label = round(value, 1)), 
            position = position_dodge(0.8), 
            vjust = -0.5, 
            size = 3.5, 
            fontface = "bold") +
  
  # הגריד המפריד: שורות=מדד, עמודות=שנת טסט+איבר גוף
  facet_grid(Metric_Type ~ Target_Label + Body_Part, scales = "free_y", switch = "y") +
  
  # צבעים
  scale_fill_manual(values = c("Native (Same Year)" = "#2a9d8f", 
                               "External (Cross-Year)" = "#e76f51")) +
  
  # הרחבת ציר Y למנוע חיתוך
  scale_y_continuous(expand = expansion(mult = c(0, 0.35))) + 
  coord_cartesian(clip = "off") +
  
  labs(
    title = "The Generalization Gap",
    subtitle = "Performance comparison across seasons (Separated by Test Year)",
    y = NULL,
    x = "Source Training Year",
    fill = "Condition"
  ) +
  
  theme_bw(base_size = 12) +
  theme(
    legend.position = "top",
    legend.title = element_blank(),
    strip.background = element_rect(fill = "#f0f0f0"),
    strip.text = element_text(face = "bold", size = 10),
    strip.placement = "outside",
    axis.text.x = element_text(face = "bold", size = 11),
    panel.spacing = unit(1.5, "lines")
  )

print(p_gap)

