# Build a reference for validating the Python port.
# For each sex: impute once exactly as analyze_motherhood.R does (bs = F),
# save the imputed data, and save R pstratreg estimates for all settings.
suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(Amelia); library(pstratreg)
})
args <- commandArgs(TRUE)
infile <- args[1]; outdir <- args[2]
d_prepared <- readRDS(infile)
set.seed(14850)

impute <- function(data) {
  amelia.out <- amelia(
    x = data %>% select(-year_1, -year_2, -age_at_birth, -age_2, -sex, -birth_year),
    m = 1, idvars = c("PUBID", "w"), noms = c("race", "marital", "fulltime"),
    ords = "educ", boot.type = "none", p2s = 0
  )
  data_imp <- amelia.out$imputations$imp1 %>% mutate(wage = data$wage)
  for (varname in c("age_1", "wage_baseline", "tenure", "experience")) {
    r <- range(data[[varname]], na.rm = TRUE)
    x <- data_imp[[varname]]
    x[x < r[1]] <- r[1]; x[x > r[2]] <- r[2]
    data_imp[[varname]] <- x
  }
  data_imp
}

rhs <- "(race + poly(age_1,2) + educ + marital + fulltime + log(tenure + 1) + log(experience + 1) + wage_baseline + employed_baseline)"
settings <- tibble(
  sex = c("Men", "Men", "Women", "Women", "Women", "Women"),
  monotonicity = c(F, F, F, F, T, T),
  mean_dominance = c(F, T, F, T, F, T)
)

out <- list(); sub <- list()
for (s in c("Men", "Women")) {
  di <- impute(d_prepared |> filter(sex == s))
  write.csv(di |> mutate(educ = as.integer(educ)), file.path(outdir, paste0("imputed_", s, ".csv")), row.names = FALSE)

  # employment-only by baseline-wage subgroup (gaussian glm = LPM, as in the paper code)
  dg <- di |> mutate(wage_baseline_categories = factor(
    case_when(!employed ~ 1, wage_baseline < log(15) ~ 2, wage_baseline < log(20) ~ 3, TRUE ~ 4),
    levels = 1:4, labels = c("Not employed", "Below $15", "$15 to $20", "Over $20")))
  fit <- glm(as.formula(paste("employed ~ treated*", rhs)), data = dg, weights = dg$w)
  tr <- dg |> filter(treated)
  tr$effect_s <- predict(fit, newdata = tr, type = "response") - predict(fit, newdata = tr |> mutate(treated = FALSE), type = "response")
  sub[[s]] <- tr |> group_by(wage_baseline_categories) |>
    summarize(effect_s = weighted.mean(effect_s, w), .groups = "drop") |> mutate(sex = s)

  for (i in which(settings$sex == s)) {
    a <- settings[i, ]
    t0 <- Sys.time()
    res <- pstratreg(
      formula_y = as.formula(paste("wage ~ treated*", rhs)),
      formula_s = as.formula(paste("employed ~ treated*", rhs)),
      formula_sq_resid = as.formula(paste("~ ", gsub("^\\(", "(treated + ", rhs))),
      data = di, weights = di$w, treatment_name = "treated", homoskedastic = FALSE,
      monotonicity_negative = a$monotonicity,
      mean_dominance_y0_positive = a$mean_dominance, mean_dominance_y1_positive = a$mean_dominance,
      aggregate = TRUE, group_vars = "treated")
    el <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    r <- res$estimates_s |> filter(treated) |> left_join(res$estimates_y |> filter(treated), by = "treated") |> select(-treated)
    out[[i]] <- r |> pivot_longer(everything(), names_to = "estimand", values_to = "estimate") |>
      bind_cols(a) |> mutate(seconds = el)
  }
}
write.csv(bind_rows(out), file.path(outdir, "r_estimates.csv"), row.names = FALSE)
write.csv(bind_rows(sub), file.path(outdir, "r_subgroups.csv"), row.names = FALSE)
print(bind_rows(out), n = 100)
print(bind_rows(sub))
