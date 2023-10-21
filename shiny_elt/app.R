library(sodium)
library(shinyauthr)
library(shiny)
library(shinydashboard)
library(shinyjs)

library(arrow)
library(dplyr)
library(ggplot2)
library(ggthemes)
library(purrr)
library(tibble)
library(tidyr)

library(MASS)
library(lightgbm)
library(xgboost)
library(broom)

# 1. Auxiliar function to run model

X_train <- arrow::read_parquet("X_train_input.parquet")
y_train <- arrow::read_parquet("y_train.parquet")[["excess_seconds_flying"]]

invoke_model <- function(model_name, ...) {
  
  args <- list(...)
  
  if (model_name == "Linear Regression") {
    
    model <- lm(y ~ ., 
                data = X_train %>% mutate(y = y_train))
    feat_imp <- tidy(model) %>% dplyr::select(term, estimate) %>% filter(term != "(Intercept)")
    rmse <- sqrt(mean(model$residuals^2))
    
  } else if (model_name == "XGBoost") {
    
    model <- xgboost(data = as.matrix(X_train), label = as.matrix(y_train), 
                     objective = "reg:squarederror", nrounds = 2,
                     learning.rate = args[["learning_rate"]], num.leaves = args[["num_leaves"]],
                     n.estimators = args[["n_estimators"]])
    
    feat_imp <- as_tibble(xgb.importance(model = model) %>% dplyr::select(Feature, Gain))
    rmse <- max(model$evaluation_log[["train_rmse"]])
  
  } else if (model_name == "LightGBM") {
    
    train_data <- lightgbm::lgb.Dataset(data = as.matrix(X_train), label = as.numeric(y_train))
    params <- list(
      objective = "regression",    
      metric = "l2",
      num_leaves = args[["num_leaves"]], learning_rate = args[["learning_rate"]],
      max_depth = args[["max_depth"]],  n_estimators = args[["n_estimators"]]
    )
    
    num_rounds <- 100
    model <- lightgbm::lgb.train(params = params, data = train_data, nrounds = num_rounds)
    
    feat_imp <- as_tibble(lgb.importance(model = model) %>% dplyr::select(Feature, Gain))
    
    y_pred <- predict(model, as.matrix(X_train))
    rmse <- sqrt(mean((y_train - y_pred)^2))
    
  } else if (model_name == "Ridge") {
    
    model <- lm.ridge(y ~ ., 
                      data = X_train %>% mutate(y = y_train), lambda = args[["lambda"]])
    y_pred <- as.matrix(X_train) %*% coef(model)[2:length(coef(model))] + coef(model)[1]
    
    feat_imp <- tibble(Feature = names(coef(model)),
                       Importance = as.numeric(coef(model))) %>% 
      filter(Feature != "")
    
    rmse <- sqrt(mean((y_train - y_pred)^2))
    
  }
  
  return (list(rmse, feat_imp))
}
# asd <- invoke_model("Linear Regression")

# 2. Descritivas

desc_variables <- list(Binary = c('tcr', 'tcp', 'is_forecast', 'runway_number'),
                       Discrete = c('runway_length', 'elevation', 'esperas'),
                       Continuous = c('days_to_holiday', 'distance_from_airports', 'metar_overall_score', 'metar_wind_score', 'metar_visibility_score', 'metar_cloud_cover_score', 'metar_dew_point_spread_score', 'metar_altimeter_setting_score', 'metar_temperature_score', 'temperature', 'dew_point', 'visibility', 'pressure', 'wind_direction', 'flight_direction', 'wind_speed', 'flight_wind_direction', 'flight_wind_speed', 'number_flights_arriving', 'number_flights_departing'))

aux_plot_binary <- function (feature) {
  if (all(sort(unique(X_train[[feature]])) == c(0, 1))) {
    X_train <- X_train %>% mutate(feature := ifelse(!!sym(feature) == 0, "False", "True"))
  }
  X_train %>% 
    count(!!sym(feature)) %>% mutate(Freq = n / sum(n)) %>% 
    ggplot(aes(x = factor(.data[[feature]]), y = Freq)) + 
    geom_col() +
    scale_y_continuous(labels = scales::percent_format(scale = 100)) +
    ggthemes::theme_fivethirtyeight() +
    labs(title = glue::glue("Relative frequency of \n {feature}"), x = feature, y = "Frequency")
}
aux_plot_discrete <- function (feature) {
  X_train %>% 
    ggplot(aes(x = factor(.data[[feature]]))) +
    geom_bar() +
    ggthemes::theme_fivethirtyeight() +
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
    labs(title = glue::glue("Absolute frequency of \n {feature}"), x = feature, y = "Frequency")
}
aux_plot_continuous <- function (feature) {
  X_train %>% 
    ggplot(aes(x = .data[[feature]])) +
    geom_histogram() +
    ggthemes::theme_fivethirtyeight() +
    labs(title = glue::glue("Histogram of \n {feature}"), x = feature, y = "Frequency")
}

aux_plot_combined <- function (continuous_var, binary_var, plot_type) {
  print(plot_type)
  g <- ggplot(X_train)
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
  g + labs(title = glue::glue("Plot of {continuous_var} according to {binary_var}")) +
    ggthemes::theme_fivethirtyeight()
}
# aux_plot_combined("temperature", "is_forecast", "Violin")
# 0.0. UI ----

ui <- dashboardPage(
  
  #theme = bs_theme(version = 4, bootswatch = "minty"),
  skin = "blue",
  
  ## 0.1. Header ----
  dashboardHeader(
    title = "Cloud9",
    titleWidth = 350,
    tags$li(
      class = "dropdown",
      style = "padding: 8px;",
      logoutUI("logout")
    )
  ),
  
  ## 0.2. Sidebar ----
  dashboardSidebar(
    
    useShinyjs(),
    collapsed = FALSE,
    uiOutput("sidebar")
    
  ),
  ## 0.3. Body ----
  dashboardBody(
    # loginUI(
    #   "login",
    #   #cookie_expiry = 7
    # ),
    tabsetPanel(
      type = "tabs",
      id = "tab_selected",
      tabPanel(title = "Modelos"),
      tabPanel(title = "Descritivas")
    ),
    
    uiOutput("body")
    
  )
)

# 2. Server ----


plot_feat_imp <- function (model_name, feat_imp) {
  
  colnames(feat_imp) <- c("Feature", "Importance")
  feat_imp <- feat_imp %>% arrange(desc(Importance))
  
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
    geom_col() +
    coord_flip() +
    ggthemes::theme_fivethirtyeight()
  
  if (!(model_name %in% c("Linear Regression", "Ridge"))) {
    g <- g + scale_y_continuous(labels = scales::percent_format(scale = 100)) +
      labs(title = "Features by importance", x = "Feature", y = "Importance (%)")
  } else {
    g <- g + labs(title = "Absolute impact of features", x = "Feature", y = "Impact")
  }
  
  return(g)
}

# login set up and authentication
auth_fun <- function () {
  
  # 1.0. Login setup ----
  
  user_base <- tibble(
    user = c("DSC2023"),
    password = c("ITA-ICEA-LATAM"),
    password_hash = sapply(c("ITA-ICEA-LATAM"), sodium::password_store),
    permissions = c("standard"),
    name = c("Organizadores")
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
    
    if (credentials()$user_auth) {
      showTab(inputId = "tab_selected", target = "Modelos")
    }
  })
  
  return(credentials)
  
  
}

server <- function(input, output, session) {
  
  # credentials <- auth_fun()
  
  # 2.0. User Inputs ----
  
  ## 2.1. Models ----
  output$which_model <- renderUI({
    selectInput(
      inputId = "which_model",
      label = "Which model?",
      choices = c("Linear Regression", "XGBoost", "LightGBM"),
      selected = "LightGBM"
    )
  })
  
  ## 2.2. Learning Rate ----
  output$learning_rate <- renderUI({
    req(input$which_model)
    
    if(input$which_model == "Linear Regression"){
      NULL
    } else if (input$which_model == "Ridge") {
      div(
        numericInput(inputId = "lambda", label = "Lambda", value = 0.1, step = 0.01, min = 0, max = 1),
      )
    } else if (input$which_model == "XGBoost") { # I'm having some trobule with XGBoost's max depth
      div(
        numericInput(inputId = "learning_rate", label = "Learning Rate", value = 0.1, step = 0.01),
        numericInput(inputId = "num_leaves", label = "Maximum number of leaves", value = 31, step = 1),
        numericInput(inputId = "n_estimators", label = "Number of trees to fit", value = 100, step = 1)
      )
    } else {
      div(
        numericInput(inputId = "learning_rate", label = "Learning Rate", value = 0.1, step = 0.01),
        numericInput(inputId = "num_leaves", label = "Maximum number of leaves", value = 31, step = 1),
        numericInput(inputId = "max_depth", label = "Maximum depth of tree (negative for no limit)", value = -1, step = 1, min = -1),
        numericInput(inputId = "n_estimators", label = "Number of trees to fit", value = 100, step = 1)
      )
    }
  })
  
  
  ## 2.3. Run Model ----
  output$run_model <- renderUI({
    actionButton(inputId = "run_model",
                 label = "Run Model"
    )
  })
  
  
  # 3.0. UI Sidebar Output ----
  output$sidebar <- renderUI({
    # req(credentials()$user_auth, input$tab_selected)
    
    if( input$tab_selected == "Modelos" ){
      div(
        br(),
        uiOutput("which_model"),
        # uiOutput("additional_params"),
        uiOutput("learning_rate"),
        uiOutput("num_leaves"),
        uiOutput("max_depth"),
        uiOutput("n_estimators"),
        uiOutput("lambda"),
        uiOutput("run_model")
      )}
  })
  
  output$body <- renderUI({
    
    if( input$tab_selected == "Modelos" ) {
      column(width = 12, align = "center",
             column(width = 6, align = "center", tableOutput("table_rmse")),
             column(width = 6, align = "center", plotOutput("plot_feat_imp"))
      )
    } else if (input$tab_selected == "Descritivas") {
      
      column(width = 12, align = "center",
             br(),
             
             column(width = 4, align = "center",

                    fluidRow(selectInput(inputId = "select_continuous_var", label = "Plot of continuous variable",
                                         choices = desc_variables$Continuous, selected = "temperature"),
                             plotOutput("plot_single_cont_var", height = "270px")),

                    fluidRow(div(h4("Grouped plot"),
                                  selectInput(inputId = "comb_plot_continuous", label = "Choose a continuous variable", choices = desc_variables$Continuous),
                                  selectInput(inputId = "comb_plot_binary", label = "Choose a binary variable", choices = desc_variables$Binary),
                                  selectInput(inputId = "comb_plot_type", label = "Choose the plot's type", choices = c("Histogram", "Violin", "Boxplot"))
                                    ))),

             column(width = 4, align = "center",

                    fluidRow(selectInput(inputId = "select_discrete_var", label = "Plot of discrete variable",
                                         choices = desc_variables$Discrete, selected = "elevation"),
                             plotOutput("plot_single_disc_var", height = "270px")),

                    plotOutput("plot_combined_var")
                    
                    ),

             column(width = 4, align = "center",

                    selectInput(inputId = "select_binary_var", label = "Plot of binary variable",
                                         choices = desc_variables$Binary, selected = "tcr"),
                             plotOutput("plot_single_bin_var", height = "270px")
             )
      )
    }
    
    
  })
  
  # 4.0 UI Body
  rmse_table_data <- reactiveVal(tibble(Model = character(0),
                                        Name = character(0), 
                                        `Learning Rate` = numeric(0),
                                        `Num Leaves` = numeric(0),
                                        `Max Depth` = numeric(0),
                                        `Num Estimators` = numeric(0),
                                        RMSE = numeric(0)))
  observeEvent(input$run_model, {
    
    possible_params <- c("learning_rate", "num_leaves", "max_depth", "n_estimators", "lambda")
    args_model <- list()
    for (i in 1:length(possible_params)) {
      if (!(is.null(pluck(input, possible_params[i])))) {
        args_model[possible_params[i]] <- pluck(input, possible_params[i])
      }
    }
    args_model["model_name"] <- input$which_model
    
    dynamic_model_obj <- reactive({do.call(invoke_model, args_model)})
    model_list_values <- dynamic_model_obj()
    
    model_rmse <- model_list_values[[1]]
    model_feat_imp_plot <- plot_feat_imp(input$which_model, model_list_values[[2]])
    
    if (input$which_model == "Linear Regression") {
      learning_rate <- NA
      num_leaves <- NA
      max_depth <- NA
      num_estimators <- NA
    } else if (input$which_model == "XGBoost") {
      learning_rate <- input$learning_rate
      num_leaves <- input$num_leaves
      max_depth <- NA
      num_estimators <- input$n_estimators
    } else if (input$which_model == "LightGBM") {
      learning_rate <- input$learning_rate
      num_leaves <- input$num_leaves
      max_depth <- input$max_depth
      num_estimators <- input$n_estimators
    }
      
    # RMSE table
    rmse_table_data() %>% 
      mutate(Model = "Previous") %>% 
      add_row(Model = "Current",
              Name = input$which_model, 
              `Learning Rate` = learning_rate,
              `Num Leaves` = num_leaves,
              `Max Depth` = max_depth,
              `Num Estimators` = num_estimators,
              RMSE = model_rmse,
              .before = 1) %>% 
      rmse_table_data()
    output$table_rmse <- renderTable({rmse_table_data()})
    
    output$plot_feat_imp <- renderPlot(model_feat_imp_plot)
    
  })
  
  # plots in the section 'Descritive'
  # select_continuous_var <- reactive({input$select_continuous_var})
  output$plot_single_cont_var <- renderPlot(aux_plot_continuous(input$select_continuous_var), height = 250)
  output$plot_single_bin_var <- renderPlot(aux_plot_binary(input$select_binary_var), height = 250)
  output$plot_single_disc_var <- renderPlot(aux_plot_discrete(input$select_discrete_var), height = 250)
  
  output$plot_combined_var <- renderPlot(aux_plot_combined(input$comb_plot_continuous,
                                                           input$comb_plot_binary,
                                                           input$comb_plot_type), width = 650)
  
}

shinyApp(ui = ui, server = server)

