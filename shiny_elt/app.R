library(shinyauthr)
library(shiny)
library(shinydashboard)
library(shinyjs)
library(tidyverse)

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
    
    collapsed = TRUE,
    
    uiOutput("sidebar")
    
  ),
  ## 0.3. Body ----
  dashboardBody(
    loginUI(
      "login", 
      #cookie_expiry = 7
    ),
    tabsetPanel(
      type = "tabs",
      id = "tab_selected",
      tabPanel(
        title = "Modelos"
      )
    )
  )
)

# Define server logic required to draw a histogram
server <- function(input, output, session) {

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
  
  # 2.0. User Inputs ----
  
  ## 2.1. Models ----
  output$which_model <- renderUI({
    selectInput(
      inputId = "which_model",
      label = "Qual Modelo?",
      choices = c("Regressão Linear", "Random Forest", "XGBoost", "LightGBM", "CatBoost"),
      selected = "LightGBM"
    )
  })
  
  ## 2.2. Learning Rate ----
  output$learning_rate <- renderUI({
    req(input$which_model)
    
    if(input$which_model %in% c("Regressão Linear", "Random Forest")){
      NULL
    } else {
      numericInput(
        inputId = "learning_rate",
        label = "Learning Rate",
        value = 0.1,
        step = 0.01
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
      
      div(
        br(),
        uiOutput("which_model"),
        uiOutput("learning_rate"),
        uiOutput("run_model")
      )
      
    } 
  })
  
}

# Run the application 
shinyApp(ui = ui, server = server)
