rm(list = ls(all=TRUE))
library("FactoMineR") # For CA
library("factoextra") # For fviz_ca_biplot
​
# Set working directory that contains AUC tables
# ------
setwd('/home/saffeldt/Projects/Projects_students/theseLouis/customer_churn_2020/results/AUC_tables')
​
# Set the data names for colnames selection
# ------
dataNames_list = c('Bank', 'C2C', 'DSN', 'HR', 'K2009', 
                   'KKBox', 'Member', 'Mobile', 'SATO', 
                   'TelC', 'TelE', 'UCI', 'News')
​
# Set biplot parameters
# ------
# ------
# Possible choices
# >> noSampling_SMOTE_NCR
# >> noSampling_SMOTEru_SMOTEtk_SMOTEncr
plot_type = "noSampling_SMOTE_NCR" 
# ------
# ------
plot_param = list()
if(plot_type == "noSampling_SMOTE_NCR"){
  print(paste0("# Plotting ", plot_type))
  
  # Biplot NoSampling / SMOTE / NCR
  plot_param = list( title = 'No Sampling / SMOTE / NCR',
                     dataFile_list = c('churn_AUC_noSampling.csv', 
                                       'churn_AUC_SMOTE.csv', 
                                       'churn_AUC_NCR.csv'),
                     dataSfx_list = c('','-st','-ncr'))
  
} else if(plot_type == "noSampling_SMOTEru_SMOTEtk_SMOTEncr"){
  print(paste0("# Plotting ", plot_type))
  
  # Biplot NoSampling / SMOTEru / SMOTE Tomek / SMOTE NCR
  plot_param = list( title = 'No Sampling / SMOTE Tomek / SMOTE NCR',
                     dataFile_list = c('churn_AUC_noSampling.csv', 
                                       'churn_AUC_SMOTEru.csv', 
                                       'churn_AUC_SMOTEtomek.csv',
                                       'churn_AUC_SMOTEncr.csv'),
                     dataSfx_list = c('','-sru', '-stk','-sncr'))
} else {
  print("# Unknown plot type")
}
​
# Aggregate AUC values
# ------
myData = data.frame()
​
for(iCount in c(1:length(plot_param[["dataFile_list"]]))){
  # Load table
  tmp_data <- read.csv(plot_param[["dataFile_list"]][iCount], 
                       header = TRUE, row.names = 1)
  
  # Remove Max-Min row
  tmp_data <- tmp_data[-which(row.names(tmp_data) == 'Max-Min'),]
  
  # Keep only the columns corresponding to datasets
  tmp_data <- tmp_data[, dataNames_list]
  
  # Add suffix to row names
  row.names(tmp_data) <- paste(row.names(tmp_data), 
                               plot_param[["dataSfx_list"]][iCount], 
                               sep = '')
  
  if(nrow(myData) == 0){
    myData = tmp_data
  } else {
    myData = rbind.data.frame(myData, tmp_data)
  }
}
# Save aggregated AUC data
# ------
write.csv(myData, file = paste(plot_type, 'csv', sep = '.'))
​
# Perform AFC
# ------
res.ca <- CA (myData, graph = FALSE)
​
# Do CA Biplot
# ------
png(paste(plot_type, "png", sep = '.'), 
    units="in", width=11, height=8, res=200)
fviz_ca_biplot (res.ca, repel = TRUE
                , col.col = "blue"
                , col.row = "contrib"
                #, title = plot_param[["title"]]
                , title = ''
                , gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07")
)
dev.off()