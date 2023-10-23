library(bslib)
library(DT)
library(sodium)
library(shinyauthr)
library(shiny)
library(shinydashboard)
library(shinyjs)
library(waiter)

library(arrow)
library(dplyr)
library(ggplot2)
library(plotly)
library(ggthemes)
library(purrr)
library(tibble)
library(tidyr)

library(MASS)
library(lightgbm)
library(xgboost)
library(rpart)
library(broom)

source("99_functions.R")


# 1. Auxiliar function to run model

X_train <- arrow::read_parquet("data/pseudoX_train.parquet")
y_train <- arrow::read_parquet("data/pseudoy_train.parquet")[["excess_seconds_flying"]]

X_test <- arrow::read_parquet("data/pseudoX_test.parquet")
y_test <- arrow::read_parquet("data/pseudoy_test.parquet")[["excess_seconds_flying"]]

invoke_model <- function(model_name, ...) {
  
  args <- list(...)
  
  if (model_name == "Linear Regression") {
    
     model <- lm(y ~ ., 
                 data = X_train %>% mutate(y = y_train))
     
     feat_imp <- tidy(model) %>% dplyr::select(term, estimate) %>% filter(term != "(Intercept)")
     
     y_pred <- predict(model, X_test)
     rmse <- sqrt(mean((y_pred - y_test)^2))
    
  } else if (model_name == "XGBoost") {
    
    model <- xgboost(data = as.matrix(X_train), label = as.matrix(y_train), 
                     objective = "reg:squarederror", nrounds = 2,
                     learning.rate = args[["learning_rate"]], num.leaves = args[["num_leaves"]],
                     n.estimators = args[["n_estimators"]])
    feat_imp <- as_tibble(xgb.importance(model = model) %>% dplyr::select(Feature, Gain))
    
    y_pred <- predict(model, as.matrix(X_test))
    rmse <- sqrt(mean((y_pred - y_test)^2))
    
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
    
    y_pred <- predict(model, as.matrix(X_test))
    rmse <- sqrt(mean((y_test - y_pred)^2))
    
  } else if (model_name == "Ridge Regression") {
    
    X_train_scaled <- X_train %>% 
      mutate(across(
        where(is.numeric), 
        ~ (.-mean(.))/sd(.)
      ))
    
    X_test_scaled <- X_test %>% 
      mutate(across(
        where(is.numeric), 
        ~ (.-mean(.))/sd(.)
      ))
    
    model <- lm.ridge(y ~ ., 
                      data = X_train_scaled %>% mutate(y = y_train), lambda = args[["lambda"]])
    
    feat_imp <- tibble(Feature = names(coef(model)),
                       Importance = as.numeric(coef(model))) %>% 
      filter(Feature != "")
    
    y_pred <- as.matrix(cbind(const=1,X_test_scaled)) %*% coef(model)
    rmse <- sqrt(mean((y_test - y_pred)^2))
    
  } else if (model_name == "Decision Tree") {
    
    model <- rpart(y ~ ., data = X_train %>% mutate(y = y_train), method = "anova",
                   control = rpart.control(maxdepth=args$max_depth, cp=0))
    
    y_pred <- predict(model, newdata = X_test)
    rmse <- sqrt(mean((y_test - y_pred)^2))
     
    feat_imp <- tibble(Feature = names(model$variable.importance),
                       Importance = as.numeric(model$variable.importance))
    
  }
  
  return (list(rmse, feat_imp))
}

# 2. Descritivas
desc_variables <- list(Binary = c(
                         'tcr', 'tcp', 'is_forecast', 'destino_SBBR', 'destino_SBCF', 
                         'destino_SBCT', 'destino_SBFL', 'destino_SBGL', 'destino_SBGR', 
                         'destino_SBKP', 'destino_SBPA', 'destino_SBRF', 'destino_SBRJ', 
                         'destino_SBSP', 'destino_SBSV'),
                       Continuous = c(
                         'days_to_holiday', 'distance_from_airports', 'metar_overall_score', 
                         'metar_wind_score', 'metar_visibility_score', 'metar_cloud_cover_score', 
                         'metar_dew_point_spread_score', 'metar_altimeter_setting_score', 
                         'metar_temperature_score', 'temperature', 'dew_point', 'visibility', 
                         'pressure', 'wind_direction', 'flight_direction', 'wind_speed', 
                         'flight_wind_direction', 'flight_wind_speed', 'number_flights_arriving', 
                         'number_flights_departing', 'minute_sin', 'minute_cos', 'hour_sin', 
                         'hour_cos', 'day_sin', 'day_cos', 'week_sin', 'week_cos', 'month_sin',
                         'month_cos', 'runway_length', 'elevation', 'esperas', 'runway_number'))

names(desc_variables$Binary) <- map_chr(desc_variables$Binary, snake_to_clean_names)
names(desc_variables$Continuous) <- map_chr(desc_variables$Continuous, snake_to_clean_names)

desc_variables$Binary <- desc_variables$Binary[order(names(desc_variables$Binary))]
desc_variables$Continuous <- desc_variables$Continuous[order(names(desc_variables$Continuous))]


# Getting desc stats for variables ----

table_stats <- bind_rows(map_dfr(desc_variables$Binary, \(x) get_stats(X_train, x, 'Binary')),
                         map_dfr(desc_variables$Continuous, \(x) get_stats(X_train, x, 'Continuous')))

# 0.0. UI ----

ui <- dashboardPage(
  
  skin = "blue",
  
  ## 0.1. Header ----
  dashboardHeader(
    title = "Cloud9  -  ITA DSC 2023",
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
    theme = bs_theme(version = 4, bootswatch = "united"),
    id = "dashboard_body",
    tags$head(tags$style(HTML("
      .content-wrapper {
        min-height: 150vh !important;
        background-color: #f5f5f5 !important; 
      }
      
      table.dataTable thead th {
      background-color: darkblue;
      color: white;
    }
    "))),
    
    autoWaiter(),
    
    loginUI(
      "login",
      #cookie_expiry = 7
    ),
    tabsetPanel(
      type = "tabs",
      id = "tab_selected",
      tabPanel(title = "Modelos"),
      tabPanel(title = "Descritivas"),
      tabPanel(title = "Tabelas")
    ),
    
    uiOutput("body")
    
  )
)

# 2. Server ----

server <- function(input, output, session) {
  
  waiter_obj <- Waiter$new(id = "dashboard_body",
                           html = tagList(spin_ellipsis(), h4("Fitting model. This sould take some seconds.")))
  
  credentials <- auth_fun()
  
  # 2.0. User Inputs ----
  
  ## 2.1. Models ----
  output$which_model <- renderUI({
    selectInput(
      inputId = "which_model",
      label = "Which model?",
      choices = c("Linear Regression", "Ridge Regression", "XGBoost", "LightGBM", "Decision Tree"),
      selected = "LightGBM"
    )
  })
  
  ## 2.2. Learning Rate ----
  output$learning_rate <- renderUI({
    req(input$which_model)
    
    if (is.null(input$which_model)) {
      return(NULL)
    }
    
    if(input$which_model %in% c("Linear Regression")) {
      NULL
    } else if (input$which_model == "Decision Tree"){
      div(
        numericInput(inputId = "max_depth", label = "Maximum depth of tree", value = 6, step = 1, min = 2),
      )
    } else if (input$which_model == "Ridge Regression") {
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
    
    req(credentials()$user_auth, input$tab_selected)
    
    if( input$tab_selected == "Modelos" ){
      div(align = "center",
          br(),
          uiOutput("which_model"),
          # uiOutput("additional_params"),
          uiOutput("learning_rate"),
          uiOutput("num_leaves"),
          uiOutput("max_depth"),
          uiOutput("n_estimators"),
          uiOutput("lambda"),
          uiOutput("run_model"),
          br(),
          div(style = "font-size: 10px;",
              "Models should take just some seconds to run, but higher values for the hyperparameters will naturally increase the required time.")
      )
      
    }
  })
  
  output$body <- renderUI({
    
    if (is.null(input$tab_selected)) {
      return(NULL)
    }
    
    if( input$tab_selected == "Modelos" ) {
      column(width = 12, align = "center",
             column(width = 12, align = "center", DTOutput("table_rmse")),
             column(width = 12, align = "center", plotlyOutput("plot_feat_imp"))
      )
    } else if (input$tab_selected == "Descritivas") {
      column(width = 12, align = "center",
             br(),
             
             column(width = 6, align = "center",
                    
                    fluidRow(selectInput(inputId = "select_continuous_var", label = "Plot of continuous variable",
                                         choices = desc_variables$Continuous, selected = "temperature"),
                             plotlyOutput("plot_single_cont_var", height = "270px")),
                    
                    fluidRow(div(h4("Grouped plot"),
                                 selectInput(inputId = "comb_plot_continuous", label = "Choose a continuous variable", choices = desc_variables$Continuous),
                                 selectInput(inputId = "comb_plot_binary", label = "Choose a binary variable", choices = desc_variables$Binary),
                                 selectInput(inputId = "comb_plot_type", label = "Choose the plot's type", choices = c("Histogram", "Violin", "Boxplot"))
                    ))),
             
             column(width = 6, align = "center",
                    
                    selectInput(inputId = "select_binary_var", label = "Plot of binary variable",
                                choices = desc_variables$Binary, selected = "tcr"),
                    plotlyOutput("plot_single_bin_var", height = "270px")
             )
      )
    } else {
      
      div(align = "center",
          h2("Stats", align = "center"),
          DTOutput("table_stats")
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
    
    waiter_obj$show()
    
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
    
    waiter_obj$hide()
    
    if (input$which_model == "Linear Regression") {
      learning_rate <- NA
      num_leaves <- NA
      max_depth <- NA
      num_estimators <- NA
    } else if (input$which_model == "Ridge Regression"){
      learning_rate <- input$lambda
      num_leaves <- NA
      max_depth <- NA
      num_estimators <- NA
    } else if (input$which_model == "Decision Tree"){
      learning_rate <- NA
      num_leaves <- NA
      max_depth <- input$max_depth
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
              RMSE = round(model_rmse, 2),
              .before = 1) %>% 
      rmse_table_data()
    output$table_rmse <- renderDT({rmse_table_data()})
    
    output$plot_feat_imp <- renderPlotly(model_feat_imp_plot)
    
  })
  
  # plots and stats in the section 'Descritive'
  output$table_stats <- renderDT({
    table_stats %>% 
      mutate(across(where(is.numeric), ~round(.x, 2))) %>% 
      mutate(Feature = sapply(Feature, snake_to_clean_names)) %>% 
      arrange(desc(Feature))
  })
  
  
  # select_continuous_var <- reactive({input$select_continuous_var})
  output$plot_single_cont_var <- renderPlotly(aux_plot_continuous(X_train, input$select_continuous_var))
  output$plot_single_bin_var <- renderPlotly(aux_plot_binary(X_train, input$select_binary_var))
  
  output$plot_combined_var <- renderPlotly(aux_plot_combined(X_train,
                                                           input$comb_plot_continuous,
                                                           input$comb_plot_binary,
                                                           input$comb_plot_type))
  
}

shinyApp(ui = ui, server = server)