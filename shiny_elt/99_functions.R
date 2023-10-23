# Plots ----
aux_plot_binary <- function (df, feature) {
  if (all(sort(unique(df[[feature]])) == c(0, 1))) {
    df <- df %>% mutate(feature := ifelse(!!sym(feature) == 0, "False", "True"))
  }
  ggplotly(df %>% 
    count(!!sym(feature)) %>% mutate(Freq = n / sum(n)) %>% 
    ggplot(aes(x = factor(.data[[feature]]), y = Freq)) + 
    geom_col(fill='darkblue') +
    scale_y_continuous(labels = scales::percent_format(scale = 100)) +
    ggthemes::theme_fivethirtyeight() +
    labs(title = glue::glue("Relative frequency of {snake_to_clean_names(feature)}"), x = feature, y = "Frequency") +
    theme(
      panel.background = element_rect(fill = "#f5f5f5", colour = NA),
      plot.background = element_rect(fill = "#f5f5f5", colour = NA),
    ), height = 250)
}
aux_plot_discrete <- function (df, feature) {
  ggplotly(df %>% 
    ggplot(aes(x = factor(.data[[feature]]))) +
    geom_bar(fill='darkblue') +
    ggthemes::theme_fivethirtyeight() +
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
    labs(title = glue::glue("Absolute frequency of {snake_to_clean_names(feature)}"), x = feature, y = "Frequency") +
    theme(
      panel.background = element_rect(fill = "#f5f5f5", colour = NA),
      plot.background = element_rect(fill = "#f5f5f5", colour = NA),
    ), height = 250)
}
aux_plot_continuous <- function (df, feature) {
  ggplotly(df %>% 
    ggplot(aes(x = .data[[feature]])) +
    geom_histogram(fill='darkblue') +
    ggthemes::theme_fivethirtyeight() +
    labs(title = glue::glue("Histogram of {snake_to_clean_names(feature)}"), x = feature, y = "Frequency") +
    theme(
      panel.background = element_rect(fill = "#f5f5f5", colour = NA),
      plot.background = element_rect(fill = "#f5f5f5", colour = NA),
    ), height = 250)
}

aux_plot_combined <- function (df, continuous_var, binary_var, plot_type) {
  print(plot_type)
  g <- ggplot(df)
  if (plot_type == "Histogram") {
    g <- g + 
      geom_histogram(aes(x = .data[[continuous_var]])) +
      facet_wrap(~.data[[binary_var]])
  } else if (plot_type == "Boxplot") {
    g <- g + 
      geom_boxplot(aes(group = factor(.data[[binary_var]]), y = .data[[continuous_var]]))
  } else if (plot_type == "Violin") {
    g <- g + 
      geom_violin(aes(y = factor(.data[[binary_var]]), x = .data[[continuous_var]]))
  }
  ggplotly(g + labs(title = glue::glue("Plot of {continuous_var} according to {binary_var}")) +
    ggthemes::theme_fivethirtyeight() +
    theme(
      panel.background = element_rect(fill = "#f5f5f5", colour = NA),
      plot.background = element_rect(fill = "#f5f5f5", colour = NA),
    ), width = 650)
}

plot_feat_imp <- function (model_name, feat_imp) {
  
  colnames(feat_imp) <- c("Feature", "Importance")
  feat_imp <- feat_imp %>% 
    mutate(Feature = sapply(Feature, snake_to_clean_names)) %>% 
    arrange(desc(Importance))
  
  if (!(model_name %in% c("Linear Regression", "Ridge"))) {
    feat_imp <- feat_imp %>% 
      mutate(Importance = Importance / sum(Importance)) %>%
      dplyr::slice(1:10)
    feat_imp <- feat_imp %>%  add_row(Feature = "Others", Importance = 1 - sum(feat_imp$Importance))
    
  } else {
    feat_imp <- feat_imp %>% dplyr::slice(1:10)
  }
  
  g <- feat_imp %>% 
    ggplot(aes(x = reorder(Feature, Importance), y = Importance)) +
    geom_col(fill="darkblue") +
    coord_flip() +
    ggthemes::theme_fivethirtyeight() +
    theme(
      panel.background = element_rect(fill = "#f5f5f5", colour = NA),
      plot.background = element_rect(fill = "#f5f5f5", colour = NA),
    )
  
  if (!(model_name %in% c("Linear Regression", "Ridge"))) {
    g <- g + scale_y_continuous(labels = scales::percent_format(scale = 100)) +
      labs(title = "Feature Importance", x = "Feature", y = "Importance (%)")
  } else {
    g <- g + labs(title = "Absolute impact of features", x = "Feature", y = "Impact")
  }
  
  return(ggplotly(g))
}

# Tables ----
get_stats <- function (df, variable, type_var) {
  table_stats <- tibble(Type = type_var,
         Feature = variable,
         `N values` = length(unique(df[[variable]])),
         Mean = mean(df[[variable]]),
         SD = sd(df[[variable]]),
         Min = min(df[[variable]]),
         Q1 = quantile(df[[variable]])[2],
         Median = median(df[[variable]]),
         Q3 = quantile(df[[variable]])[4],
         Max = max(df[[variable]]))
  
  if(type_var == 'Binary'){
    table_stats[c('Min', 'Q1', 'Median', 'Q3', 'Max')] <- NA
  }
  
  return(table_stats)
}

# Column Names ----
snake_to_clean_names <- function(old_name){
  old_new_names <- list(
    'tcr' = 'TCR',
    'tcp' = 'TCP',
    'is_forecast' = 'É Previsão',
    'destino_SBBR' = 'Destino BSB',
    'destino_SBCF' = 'Destino CNF',
    'destino_SBCT' = 'Destino CWB',
    'destino_SBFL' = 'Destino FLN',
    'destino_SBGL' = 'Destino GIG',
    'destino_SBGR' = 'Destino GRU',
    'destino_SBKP' = 'Destino VCP',
    'destino_SBPA' = 'Destino POA',
    'destino_SBRF' = 'Destino REC',
    'destino_SBRJ' = 'Destino SDU',
    'destino_SBSP' = 'Destino CGH',
    'destino_SBSV' = 'Destino SSA',
    'runway_length' = 'Comprimento da pista',
    'elevation' = 'Elevação',
    'esperas' = 'Esperas',
    'runway_number' = 'Qtd. Pistas',
    'days_to_holiday' = 'Dias Feriado',
    'distance_from_airports' = 'Distância AERO',
    'metar_overall_score' = 'Score Geral',
    'metar_wind_score' = 'Score Vento',
    'metar_visibility_score' = 'Score Visibilidade',
    'metar_cloud_cover_score' = 'Score Nuvens',
    'metar_dew_point_spread_score' = 'Score Dew Point',
    'metar_altimeter_setting_score' = 'Score Altímetro',
    'metar_temperature_score' = 'Score Temperatura',
    'temperature' = 'Temperatura',
    'dew_point' = 'Dew Point',
    'visibility' = 'Visibilidade',
    'altimeter_setting' = 'Altímetro',
    'wind_speed' = 'Velocidade Vento',
    'wind_direction' = 'Direção Vento',
    'pressure' = 'Pressão',
    'flight_direction' = 'Direção Voo',
    'flight_wind_direction' = 'Direção Vento-Voo',
    'flight_wind_speed' = 'Interação Vento-Voo',
    'number_flights_arriving' = 'Qtd. Voos Chegando',
    'number_flights_departing' = 'Qtd. Voos Partindo',
    'minute_sin' = 'Minuto (sen.)',
    'minute_cos' = 'Minuto (cos.)',
    'hour_sin' = 'Hora (sen.)',
    'hour_cos' = 'Hora (cos.)',
    'day_sin' = 'Dia (sen.)',
    'day_cos' = 'Dia (cos.)',
    'month_sin' = 'Mês (sen.)',
    'month_cos' = 'Mês (cos.)',
    'week_sin' = 'Dia Semana (sen.)',
    'week_cos' = 'Dia Semana (cos.)'
  )
  
  if(old_name %in% names(old_new_names)){
    return(old_new_names[[old_name]])
  } 
  
  return(old_name)
}

# Auth ----
auth_fun <- function () {
  
  # 1.0. Login setup ----
  
  user_base <- tibble(
    user = c("ITA-ICEA-LATAM", "1"),
    password = c("DSC2023", "1"),
    password_hash = sapply(c("ITA-ICEA-LATAM", "1"), sodium::password_store),
    permissions = c("standard", "standard"),
    name = c("Organizadores", "TESTE")
  )
  
  # call login module supplying data frame, user and password cols and reactive trigger
  credentials <- shinyauthr::loginServer(
    id = "login",
    data = user_base,
    user_col = user,
    pwd_col = password_hash,
    sodium_hashed = TRUE,
    #cookie_logins = TRUE, # TRUE if sessionif, cookie_getter is provided
    log_out = reactive(logout_init()),
    #sessionid_col = sessionid,
    #cookie_getter = get_sessionids_from_db,
    #cookie_setter = add_sessionid_to_db
  )
  
  # call the logout module with reactive trigger to hide/show
  logout_init <- shinyauthr::logoutServer(
    id = "logout",
    active = reactive(credentials()$user_auth)
  )
  
  observe({
    if (credentials()$user_auth) {
      shinyjs::removeClass(selector = "body", class = "sidebar-collapse")
    } else {
      shinyjs::addClass(selector = "body", class = "sidebar-collapse")
    }
  })
  
  user_info <- reactive({
    credentials()$info
  })
  
  observeEvent(credentials()$user_auth, {
    
    hideTab(inputId = "tab_selected", target = "Modelos")
    hideTab(inputId = "tab_selected", target = "Descritivas")
    hideTab(inputId = "tab_selected", target = "Tabelas")
    
    if (credentials()$user_auth) {
      showTab(inputId = "tab_selected", target = "Modelos")
      showTab(inputId = "tab_selected", target = "Descritivas")
      showTab(inputId = "tab_selected", target = "Tabelas")
    }
  })
  
  return(credentials)
  
  
}