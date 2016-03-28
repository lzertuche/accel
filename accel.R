

library(data.table)
library(ggplot2)
library(caret)
library(signal)
library(scales)
library(moments)
library(entropy)
library(TSA)
library(randomForest)
library(foreach)
library(doParallel)
library(xgboost)
library(kknn)



#read and concatenate all raw data, add id column, rename columns
ts <- data.table() #ts = timeseries
for (i in 1:15){
  tmp <- data.table(read.csv( paste0('./DATA/accel/', i,'.csv')))
  tmp[,id := factor(i)]
  ts <- rbindlist(list(ts,tmp), use.names = F) #replaced rbind
}
rm(tmp)

#describe action labels
# --- 1: Working at Computer, work
# --- 2: Standing Up, Walking and Going updown stairs
# --- 3: Standing, stand
# --- 4: Walking, walk
# --- 5: Going UpDown Stairs, climb
# --- 6: Walking and Talking with Someone
# --- 7: Talking while Standing



#rename columns
names(ts) <- c("nseq",'ax', 'ay', 'az', 'action','id')

###look at time window of raw data for all actions
nsamples = 500
start.sample = 250
ts.plot <- data.table()

for (i in 1:7){
  tmp<- ts[id =="11" & action ==as.character(i), ][start.sample:(start.sample+nsamples-1), ][ ,nseq := 1:nsamples]
  ts.plot <- rbindlist(list(ts.plot,tmp), use.names = T)
}
ggplot(data = na.omit(ts.plot)) + geom_line(aes(x = nseq , y = ax), color = "black", size=1) + 
                                  geom_line(aes(x = nseq , y = az), color = "red1", size =1) + 
                                  facet_grid(action ~ ., scales = "free_y") + 
                                  labs(title = "Raw Data for X and Z components", x = "Samples", y = "") +
                                  theme(axis.text.y = element_blank())


#remove rows for composite actions action==0,2,6 
ts <- ts[action != 0 & action != 2 & action != 6 ]

#make action and id into factors
ts[,action := factor(action) ]
ts[,id := factor(id) ]

#meaningful level names
levels(ts$action) <- list(work="1", stand="3", walk ="4", climb = "5", talk = "7")

# test digital filter
if(F){ 
sf <- 52 #sample freq
cf <- 3 #cutoff freq
secs <-  1
t <- seq(0, secs, len = sf*secs*2)
x <- sin(2*pi*t) #1hz
noise <-  .5*sin(2*pi*t*10) 
xnoise <- x + noise
f <- butter(7, cf/(sf/2)) #W = cutoff band, input as normalized to nyquist f
xclean <- filtfilt(f, xnoise)
ggplot() + geom_line(aes(x = t, y = x, color = "a" ), size=2)  +  
           geom_line(aes(x = t, y = noise, color = "b") , size = 1) + 
           geom_line(aes(x = t, y = xnoise, color = "c"), size = 1) +
           geom_line(aes(x = t, y = xclean, color = "d"),size = 1)  +
           scale_color_manual ("", values = c("a"="red","b"="blue", "c"="green","d"="yellow" ), 
                               label=c("Original", "HF-noise", "Orig+Noise","Filtered" ))
}

##define more time series
#accel magnitude
ts[,amag := sqrt(ax^2 + ay^2 + az^2)] 

#principal components
pc <- prcomp(ts[ ,.(ax,ay,az)], center = T, scale. = T)
ts[,pc1 := pc$x[,1]]
ts[,pc2 := pc$x[,2]]
ts[,pc3 := pc$x[,2]]
rm(pc)

#separate the signals into hf and lf
cutoffHZ <- 6 #based on ref: ACCELEROMETER SIGNAL PRE-PROCESSING...pdf
sampleHz <- 52
nyqHZ = sampleHz/2 #nyquist
f <- butter(9, cutoffHZ/nyqHZ) #9th order lpf

#create lowfreq components 
ts[,c('lax', 'lay', 'laz', 'lamag', 'lpc1', 'lpc2', 'lpc3') := 
       lapply(.(ax, ay, az, amag, pc1, pc2, pc3), function(x) (filtfilt(f, x)))]

#keep hf components
ts[, hax := ax - lax]
ts[, hay := ay - lay]
ts[, haz := az - laz]
ts[, hamag := amag -lamag]
ts[, hpc1 := pc1 - lpc1]
ts[, hpc2 := pc2 - lpc2]
ts[, hpc3 := pc3 - lpc3]

#look at hf and lf components 
if(F){
subjid ='1'
actionid = 'talk'
ggplot(data = ts[id == subjid][action == actionid][1:250]) + 
  geom_line(aes(x=nseq, y=amag), color = "red") +
  geom_line(aes(x=nseq, y=lamag), color = "blue") 

ggplot(data = ts[id == subjid][action == actionid][1:250]) + 
  geom_line(aes(x=nseq, y=hamag), color = "green")  
}

####extract features from ts ####

feature.extract <- function (dt) {
  
  #utililty functions feature extraction
  zerocross <- function (x) { return (sum(diff(sign(x)) != 0)) }
  peak2peak <- function (x) { return (max(x) - min(x)) }
  rms <- function (x) { return (sqrt(mean(x^2))) }
  
  #center and subset relevant cols
  dt[, c("lax", "lay", "laz","hax", "hay","haz","lamag", "hamag") :=
       lapply(.(lax, lay, laz, hax, hay, haz, lamag, hamag), scale, center=T, scale=F)]
  dts <- dt[ ,.(lax, lay, laz, hax, hay, haz, lamag, hamag, lpc1, lpc2, lpc3, hpc1, hpc2, hpc3)] 
  
  #names of all time series
  namevec <- names(dts)
  
  # mean
  means <- lapply(dts, mean)
  names(means) <- lapply(namevec, paste0, ".avg" )

  #stdev
  sds <- lapply(dts, sd)
  names(sds) <- lapply(namevec, paste0, ".sd" )
  
  #zero crossings
  zcs <- lapply(dts, zerocross)
  names(zcs) <- lapply(namevec, paste0, ".zc" )
  
  #minmax
  p2p <- lapply(dts, peak2peak)
  names(p2p) <- lapply(namevec, paste0, ".p2p" )
  
  #rms
  rmsvec <- lapply(dts, rms)
  names(rmsvec) <- lapply(namevec, paste0, ".rms" )
  
  #kurtosis
  kurt <- lapply(dts, kurtosis)
  names(kurt) <- lapply(namevec, paste0, ".kur" )
  
  #skew
  skew <- lapply(dts, skewness)
  names(skew) <- lapply(namevec, paste0, ".skw" )
  
  #crest factor (peak/rms)
  cfvec <-mapply(`/`, lapply(dts, max), rmsvec)  
  names(cfvec) <- lapply(namevec, paste0, ".cf" )
  
  #rms for velocity 
  rmsvec.vel <- lapply(dts, function (x) rms(diffinv(as.vector(x))) )
  names(rmsvec.vel) <- lapply(namevec, paste0, ".Vrms" )
  
  #entropy
  entr <- lapply(dts, function(x) entropy(discretize(x, numBins = 10 )))
  names(entr) <- lapply(namevec, paste0, ".ent" )
  
  #correct label
  label <- dt[1, action]
  names(label) <- "label"
  
  return ( c(zcs, p2p, rmsvec, kurt, skew, cfvec, rmsvec.vel, entr, label) )  
}

#sliding window parameters
winsecs = 1.5 #window length in seconds
winsize = sampleHz*winsecs 
overlap = .10 #percentage
ix = 1
hop = round(winsize * overlap)

#pre allocate feature objc aprox size
ffrows <- round(nrow(ts)/(winsize*overlap))
nfeat <- length( feature.extract (ts[1:(1 + winsize), ]) )

#feature dt
fdt <- data.table(matrix(data = 0.0, nrow = ffrows, ncol = nfeat) )
names(fdt) <- names( feature.extract (ts[1:(1+ winsize), ]) )

#exctract features
ptm <- proc.time() 
ix <- 1  #index for time series
rx <- 1L #index for feature dt

while ((ix) <= nrow(ts)-winsize ) {
  
  #make sure the whole window has the same action label
  if (ts[ix, action] == ts[ix + winsize, action] ) {
    
    #extract features 
    set(fdt,rx, 1:nfeat, feature.extract(ts[ix:(ix + winsize), ]) )
    rx = rx + 1L
    #move to next window
    ix <- ix + hop
   
  } else {
      ix <- ix + 1
  }
}
print("feature extraction lasted:")
(proc.time() - ptm)/60

#remove few extra rows created during prealloc
fdt <- fdt[lax.p2p != 0 & lpc1.rms != 0 & lpc2.rms != 0,]
#make label into factor for fdt
fdt[,label := factor(label) ]
levels(fdt$label) <- list(work="1", stand="2", walk ="3", climb = "4", talk = "5")
saveRDS(fdt, "features_dt")
fdt_backup <- fdt
####modeling####

#remove correlated features
corrmat <- cor(fdt[,-("label"),with = F])
highcorr <- findCorrelation(na.omit(corrmat), cutoff = 0.8, names = T, exact = T)
fdt <- fdt[,-highcorr,with=F]

#fit a RF
if (F) {
set.seed(99)
ptm <- proc.time() 
rf.fit <- randomForest(label ~  ., data = fdt, importance = T, ntree=600, nodesize = 4)
(proc.time() - ptm)/60
varImpPlot(rf.fit,n.var=40, main = "Variable Importance Plot", pch=16)
imp <- importance(rf.fit )
plabel <- predict(rf.fit)
rf.cm <- confusionMatrix(fdt$label,plabel)
}


#fit a parallel RF
registerDoParallel(cores=4)
ptm <- proc.time() 
ntree <- 1000
rf.par <- foreach(ntreepar=rep(ntree/4, 4), .combine=combine, .packages='randomForest') %dopar% {
          randomForest(label ~  ., data = fdt, importance = T, ntree=ntreepar)
}
closeAllConnections() #needed in windows machines to kill zombie R processes
(ptm - proc.time() ) / 60 
varImpPlot(rf.par, n.var= ncol(fdt) - 1)
imp <- importance(rf.par)
plabel <- predict(rf.par)
rf.restult <- confusionMatrix(fdt$label, plabel)
saveRDS(rf.par, "rfobj")


#### xgboost trees
trainix <- createDataPartition(fdt$label, p = .85,    list = FALSE, times = 1)
traindt <- fdt[trainix, ]
testdt <- fdt[-trainix, ]
xgtrain <- xgb.DMatrix(as.matrix(traindt[, -"label", with = F]), label = as.numeric(traindt$label)-1) 
xgtest <- xgb.DMatrix(as.matrix(testdt[, -"label", with = F]), label = as.numeric(testdt$label)-1 ) 
wl <- list(eval = xgtest)
#train
xgbt <- xgb.train(data = xgtrain, max.depth = 7, eta = .4 , nrounds=250, watchlist = wl, 
                  objective = "multi:softmax", num_class=5, early.stop.round = 15, 
                  verbose = 1, maximize = F)
#test 
xgpred <- predict(xgbt,xgtest)
xgpred <- as.factor(as.character(xgpred))
levels(xgpred) <- levels(fdt$label)
xg.cm <- confusionMatrix(testdt$label, xgpred)

#var imp xgb
xgb.imp <- xgb.importance(feature_names = names(fdt), model=xgbt)
#diy importance plot
nfeatures=25
ggplot(data = xgb.imp[1:nfeatures], aes(y=Gain, x=Feature)) + geom_bar(stat="identity", width = 0.5) +
                             scale_x_discrete(limits=xgb.imp[nfeatures:1,Feature]) +     
                             theme(text = element_text(size=14),axis.text = element_text(size=14)) + 
                             ggtitle("Importance plot for Gradient Boosted Tree") +
                             coord_flip() 
                           


xgb.plot.importance(xgb.imp[1:20])

#10 fold cross validation 
xgfdt <- xgb.DMatrix(as.matrix(fdt[, -"label", with = F]), label = as.numeric(fdt$label)-1) 

xgbt.cv <- xgb.cv(data = xgfdt, nfold = 10, max.depth = 7, eta = .4 , nrounds=300, 
                  objective = "multi:softmax", num_class=5, early.stop.round = 15, 
                  maximize = F)

#### weighted KNN
wknn <- kknn(label ~ . , train = traindt, test = testdt, kernel = "optimal", k=5)
wknn.cf <- confusionMatrix(wknn$fitted.values, testdt$label) 

                    








#plot accel time series by action, by id

ggplot(data = ts[id == "1", ], aes(x = nseq , y = ax[1:100]) ) + geom_line(color = "red") + facet_grid(action ~., scales = "free" )
#                                     facet_grid(action ~ ., scales = 'free')


subjid ='2'
actionid = 'work'
ggplot(data = ts[id == subjid,][action == actionid]) + geom_line(aes(x=nseq, y=ax), color = "red")  +
  geom_line(aes(x=nseq, y=ay), color = "blue") 





#### deep net
pp <- preProcess(x=traindt[, -"label", with = F], method = c("scale", "center"))
train.nn <- as.matrix(predict(pp, traindt[, -"label", with = F] ))
train.labelnn <- as.integer(traindt$label)
pp <- preProcess(x=testdt[, -"label", with = F], method = c("scale", "center"))
test.nn <- as.matrix(predict(pp, testdt[, -"label", with = F] ))
test.labelnn <- as.integer(testdt$label)

nn <- mx.mlp(train.nn, train.labelnn, hidden_node=c(30,5), out_node = 5,
                num.round=20, learning.rate=0.15, momentum=.9, dropout = 0.2, 
                eval.metric=mx.metric.accuracy)
nn.pred <- predict(nn,test.nn)
nn.pred <- max.col(t(nn.pred))
confusionMatrix(nn.pred,test.labelnn)

  
