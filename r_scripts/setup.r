install.packages("pROC", dependencies = TRUE)
install.packages("microbenchmark", dependencies = TRUE)
install.packages("gbm", dependencies = TRUE)
install.packages("xgboost", dependencies = TRUE)
devtools::install_github("Laurae2/lgbdl")
#lgbdl::lgb.dl(commit = "master",
#              compiler = "vs", # Remove this for MinGW + GPU installation
#              repo = "https://github.com/Microsoft/LightGBM", 
#              cores = 4,
#              R35 = TRUE,
#              use_gpu = TRUE)

# Error in lgbdl::lgb.dl(commit = "master", compiler = "vs", repo = "https://github.com/Microsoft/LightGBM",  : 
#  unused argument (R35 = TRUE)


lgbdl::lgb.dl(commit = "master",
              compiler = "vs", # Remove this for MinGW + GPU installation
              repo = "https://github.com/Microsoft/LightGBM", 
              cores = 4,
              use_gpu = TRUE)