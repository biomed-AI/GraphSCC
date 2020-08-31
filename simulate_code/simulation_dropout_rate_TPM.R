setwd("/data2/users/zengys/data/singlecell/number_class/splatter_simulation")

rm(list = ls())
library(splatter)
library(rhdf5)

dropout.rate <- c()
facScale=0.3
for(i in 1:2) {
  simulate <- function(nGroups=4, nGenes=2000, batchCells=2000, dropout=2) 
  {
    if (nGroups > 1) method <- 'groups'
    else             method <- 'single'
    
    group.prob <- rep(1, nGroups) / nGroups
    sim <- splatSimulate(group.prob=group.prob, nGenes=nGenes, batchCells=batchCells,
                         dropout.type="experiment", method=method,
             seed=100+i, dropout.shape=-1, dropout.mid=dropout, de.facScale=facScale)
    
    counts     <- as.data.frame(t(counts(sim)))
    truecounts <- as.data.frame(t(assays(sim)$TrueCounts))
    
    dropout    <- assays(sim)$Dropout
    mode(dropout) <- 'integer'
    
    cellinfo   <- as.data.frame(colData(sim))
    geneinfo   <- as.data.frame(rowData(sim))
    
    list(sim=sim,
         counts=counts,
         cellinfo=cellinfo,
         geneinfo=geneinfo,
         truecounts=truecounts)
  }
  
  sim <- simulate()
  
  simulation <- sim$sim
  counts <- sim$counts
  geneinfo <- sim$geneinfo
  cellinfo <- sim$cellinfo
  truecounts <- sim$truecounts
  
  dropout.rate <- c(dropout.rate, (sum(counts==0)-sum(truecounts==0))/sum(truecounts>0))
  
  X <- t(counts)
  Y <- as.integer(substring(cellinfo$Group,6))
  Y <- Y-1
  
  X = apply(X,2,function(x) (x*10^6)/sum(x)) 
  batch_1_data <- rbind(sample_labels = rep(1, length(Y)), cell_labels = Y, cluster_labels= rep(1, length(Y)), X)
  write.table(batch_1_data, file = paste("splatter_simulate_test",i,"_" ,facScale,".csv",sep=""), sep = ",", quote = F, col.names = F, row.names = T)

  
}
dropout.rate <- data.frame(dropout.rate=dropout.rate)
write.csv(dropout.rate, "simulation_dropout_rate.csv", row.names = F)
