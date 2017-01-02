# Clear up
rm(list=ls())

# Options
options(max.print = 75)

# Load in Bioconductor packages
ip <- rownames(installed.packages())
lb <- c('Biobase','BiocGenerics','parallel','AnnotationDbi','hgfocus.db','Homo.sapiens','GSE5859',
        'affy','oligo','limma','genefilter','sva')
source("https://bioconductor.org/biocLite.R")
# Check for updates
biocLite()
# Then load
for (k in lb) {
  if(k %in% ip) { library(k,character.only=T)} 
  else { print(k) }
    # biocLite(k); library(k,character.only = T)  }
}

# Load in CRAN packages
ll <- c('tidyverse','magrittr','forcats','stringr','cowplot','broom','scales','reshape2','ggrepel')
for (k in ll) {
  if(k %in% ip) { library(k,character.only=T)} 
  else { print(k) }
}

# Set up directory
dir1 <- 'C:/Users/erikinwest/Documents/bioeconometrician/github/erikdrysdale.github.io/_rmd/'
dir2 <- 'C:/Users/erikinwest/Documents/bioeconometrician/github/erikdrysdale.github.io/_rmd/extra_batch_effects/'
setwd(dir2)

# Increase the cowplot text size
theme_set(theme_cowplot(font_size=20))

############################################################
###### ----- STEP 0: CREATE MOTIVATING EXAMPLE ----- #######
############################################################

# Load data
library(GSE5859)
data(GSE5859)
# Get Expression data
geneExpression <- exprs(e)
# Find the Affymetrix control genes and drop
geneCtrl <- grepl('AFFX',rownames(e)) %>% which
geneExpression <- geneExpression[-geneCtrl,]
geneProbes <- rownames(geneExpression)
# Get the gene codes
geneCodes <- mapIds(hgfocus.db, keys=geneProbes,column="ENTREZID",keytype="PROBEID",multiVals="first")
geneSym <- mapIds(hgfocus.db, keys=geneProbes,column="SYMBOL",keytype="PROBEID",multiVals="first")
# And chromosome
geneCHR <- mapIds(Homo.sapiens, keys=geneCodes,column="TXCHROM", keytype="ENTREZID",multiVals="first") %>% str_c
# Create information table
geneAnnotation <- tibble(Probes=geneProbes,Sym=geneSym,CHR=geneCHR)

# Use genes on the Y-chromosome
y.probe <- geneAnnotation %>% filter(CHR=='chrY') %>% use_series(Probes)
y.sym <- geneAnnotation %>% filter(CHR=='chrY') %>% use_series(Sym)
yExpr <- geneExpression[which(geneProbes %in% y.probe),] %>% tbl_df %>% 
  set_colnames(factor(1:length(.))) %>% mutate(gene=y.sym) %>% gather(person,val,-gene)
# Find the gender order
sexOrder <- yExpr %>% group_by(person) %>% summarise(yExp=mean(val)) %>% arrange(yExp) %>% use_series(person)
sexSamp <- c(head(sexOrder),tail(sexOrder))
# Get a sub-sample of the data to show
ySub <- yExpr %>% filter(person %in% sexSamp)

# Create heatmap with normalized expression
gg.toyA <- ggplot(ySub %>% group_by(gene) %>% mutate(val=(val-mean(val))/sd(val)),
       aes(x=fct_relevel(person,sexSamp),y=gene,fill=val)) + 
  geom_tile() + scale_fill_gradient(low='blue',high='yellow',name='Gene Expression') +
  labs(x='Column=Person',y='Gene',subtitle='Normalized values') +
  theme(legend.position = 'bottom',axis.text.x=element_blank(),axis.ticks.x=element_blank())
# And a histogram
yHist <- yExpr %>% group_by(person) %>% summarise(val=mean(val)) %>% 
  mutate(val=(val-mean(val))/sd(val),gender=ifelse(val>0,'M','W'))
yLab <- data.frame(x=c(-0.5,0.35),y=c(2,2),label=c('Women','Men'))  
# Ploy
gg.toyB <- ggplot(yHist,aes(x=val,y=..density..)) + 
  geom_histogram(bins=36,color='black',aes(fill=gender),show.legend = F) + 
  labs(x='Mean Y-chr gene expression',y='Density',subtitle='For 208 participants') +
  geom_vline(xintercept = 0,linetype=2) +
  geom_text(data=yLab,aes(x=x,y=y,label=label,color=label),inherit.aes = F,show.legend = F,size=6)

gg.toy <- plot_grid(gg.toyA,gg.toyB,ncol=2,labels=c('A','B'))

# save_plot('toy.png',gg.toy)

#######################################################
###### ----- STEP 1: LOAD IN THE RAW DATA ----- #######
#######################################################

# # Define directory with cell files
# raw.dir <- 'C:/Users/erikinwest/Documents/Courses/EPID_823/presentation/raw'
# # Set dir
# setwd(raw.dir)
# # Read CEL files into an AffyBatch
# raw.data <- ReadAffy()
# # Use rma normalization
# norm.data <- affy::mas5(raw.data,sc=500)
# # Save for later
# setwd(dir)
# save(raw.data,norm.data,file='norm_data.RData')

# Load in RMA normalized dat
load('C:/Users/erikinwest/Documents/Courses/EPID_823/presentation/blog/norm_data.RData')
# Get expression data (and drop Affymetrix control genes)
GSE.exprs <- exprs(norm.data) %>% data.frame(rn=rownames(.),.) %>% 
  filter(!grepl('AFFX',rn)) %>% set_colnames(NULL)
# Get probe IDs
GSE.probes <- GSE.exprs[,1] %>% as.character
GSE.exprs <- GSE.exprs[,-1] %>% as.matrix %>% log2
# Get Gene Symbol and Chromosome
GSE.zid <- mapIds(hgfocus.db,keys=GSE.probes,column='ENTREZID',keytype='PROBEID',multiVals='first')
GSE.sym <- mapIds(hgfocus.db,keys=GSE.probes,column='SYMBOL',keytype='PROBEID',multiVals='first')
GSE.chr <- mapIds(Homo.sapiens,keys=GSE.zid,column='TXCHROM',keytype='ENTREZID',multiVals='first') %>% str_c
# Create information table
GSE.info <- tibble(Probes=GSE.probes,Symbols=GSE.sym,Chromosome=GSE.chr)

# Get meta-data
GSE.date <- protocolData(raw.data)@data$ScanDate %>% as.Date(format='%m/%d/%y')
GSE.gsm <- rownames(protocolData(raw.data)@data) %>% gsub('[.]CEL[.]gz','',x=.)
# Define the order so we can join it
GSE.order <- tibble(gsm=GSE.gsm,date=GSE.date)
# Load the Python data we have scraped off the web
GSE.meta <- read_csv('python_eth.csv') %>% dplyr::select(-X1) %>%
  mutate(eth=fct_recode(eth,'Chinese'='HAN CHINE','Japanese'='JAPANESE'),
         eth2=ifelse(eth=='Caucasian','Caucasian','Asian') %>% lvls_reorder(c(2,1))) %>%
  cbind(.,Date=GSE.order$date) %>% tbl_df


# Create chart for sequencing year
seqYear <- GSE.meta %>% mutate(Year=format(Date,'%Y')) %>% group_by(Year) %>%
  do(Count=table(.$eth,useNA='ifany')) %>% tidy(Count) 
gg.seqYear <- ggplot(seqYear,aes(x=Year,y=Freq,fill=Var1)) + 
  geom_bar(stat='identity',position=position_dodge(),color='black') + 
  scale_fill_discrete(name='Ethnicity') + 
  labs(y='Number of samples') + 
  theme(axis.title.x = element_blank())

####################################################
###### ----- STEP 2: GENEWISE ANALYSIS ----- #######
####################################################

# Define the ethnicity factors
ethnicity3 <- GSE.meta$eth
ethnicity2 <- GSE.meta$eth2

# Calculate Bonferroni correction (5% level) 
bonfer <- -log10(0.05/nrow(GSE.info))
# Calculate the Sidak correction
sidak <- -log10(1-(1-0.05)^(1/nrow(GSE.info)))

# Load in the Pythonic pdf scraped data
supp1 <- read_csv('supp1.csv')
# Get the mean merge data
supp1.merge <- supp1 %>% dplyr::select(affy_id,`avg(ceu)`,`avg(chb+jpt)`) %>% 
  set_colnames(c('Gene','asian.sub','white.sub'))

# Get Probe names
supp1.probe <- supp1$affy_id

# Get the moments for the two racial groups
moment.eth2 <- GSE.exprs %>% tbl_df %>% set_colnames(str_c(ethnicity2,1:ncol(.),sep='.')) %>% 
  mutate(Gene=GSE.info$Probes) %>% gather(var,val,-Gene) %>%
  separate(var,into=c('eth','num'),sep='[.]') %>% 
  group_by(Gene,eth) %>% summarise(m=mean(val),v=var(val))

# Compare the mean expression
scat.dat.eth2 <- moment.eth2 %>% dplyr::select(-v) %>% dcast(Gene~eth,value.var='m') %>%
  tbl_df %>% filter(Gene %in% supp1.probe) %>% set_colnames(c('Gene','asian.all','white.all')) %>% 
  left_join(supp1.merge,by='Gene') %>% gather(var,val,-Gene) %>% 
  separate(var,c('eth','dat'),'[.]') %>% dcast(Gene+eth~dat,value.var='val') %>% tbl_df %>%
  mutate(eth=fct_recode(eth,'Caucasian'='white','Asian'='asian') %>% lvls_reorder(c(2,1)))
# Run some Regressions
scat.lm <- scat.dat.eth2 %>% group_by(eth) %>% do(ols=lm(sub~all,data=.))
scat.lm %>% tidy(ols)
# Get the R-squared
scat.lab <- scat.lm %>% glance(ols) %>% mutate(r.squared=round(r.squared*100,1) %>% paste('R2=',.,'%',sep=''))
# Get the influence
cooksd <- scat.lm %>% augment(ols) %>% data.frame %>% mutate(probe=scat.dat.eth2$Gene) %>% tbl_df %>% arrange(desc(.cooksd))
# Get the "bad probes"
bad.probes <- cooksd %>% dplyr::select(eth,.cooksd,probe) %>% filter(.cooksd>0.01) %>% use_series(probe)


# gg it
gg.eth.scat <- ggplot(scat.dat.eth2,aes(x=sub,y=all,color=eth)) +
  geom_point(show.legend = F) + facet_wrap(~eth,ncol=2) + 
  labs(y='All data',x='Spielman data') + 
  background_grid('xy','xy') + stat_smooth(method = 'lm',se=F,color='black') +
  geom_text(data=scat.lab,inherit.aes=F,aes(x=7,y=13,color=eth,label=r.squared),show.legend=F)

#### ---- VOLCANO PLOT: SPIELMAN ---- ####

# Run row-wise ttests
spielT <- 
  rowttests(x=GSE.exprs,fac=ethnicity2) %>% cbind(GSE.info,.) %>% tbl_df %>%
  mutate(p.value=-log10(p.value)) %>% mutate(sidak=ifelse(p.value>sidak,T,F)) %>%
  filter( !(Probes %in% bad.probes) ) %>% 
  mutate(sig=ifelse(sidak & abs(dm)>0.5,T,F))

# # Find four largest expression differences
# fourT <- spielT %>% arrange(desc(abs(dm))) %>% mutate(rid=1:nrow(.)) %>% filter(rid %in% c(9,10,14,15))

# Supp table 2/3 genes
mech.genes <- c('VRK3','ATP5O','HYPK','DPAGT1')
fourT <- spielT %>% filter(Symbols %in% mech.genes)

# Get the number of statistically/biologically significant results. 
spielSig <- spielT %>% filter(sig) %>% mutate(dmp=sign(dm)) %>% count(dmp) %>%
  mutate(x=c(-1.5,2.5),y=c(45,45),label=paste(n))

# Plot
c1 <- '#F8766D'

gg.volc1 <- 
  ggplot(spielT,aes(x=dm,y=p.value)) +
  geom_point(aes(color=sig),alpha=0.5,show.legend=F) +
  geom_hline(yintercept=sidak,color=c1,linetype=2) +
  geom_vline(xintercept = c(-0.5,0.5),color=c1,linetype=2) +
  labs(x='Mean difference',y='-log10(p-value)',subtitle='Gene-wise t-test between Caucasian/Asian') +
  scale_color_manual(name='',values=c('grey',c1)) +
  geom_text_repel(data=fourT,aes(label=Symbols),nudge_y = 7,size=7) +
  geom_text(data=spielSig,inherit.aes=F,aes(x=x,y=y,label=label),color=c1,size=7) + 
  geom_text(data=data.frame(x=3,y=sidak+3),inherit.aes = F,aes(x=x,y=y,label='Sidák'),color=c1,size=7)

# Put four gene expression data into long format
long1 <- GSE.exprs[which(GSE.info$Symbols %in% fourT$Symbols),] %>% t %>%
  set_colnames(fourT$Symbols) %>% tbl_df %>%
  mutate(eth=ethnicity2,year=format(GSE.date,'%Y')) %>% gather(gene,val,-eth,-year)

long1 %>% group_by(gene,eth) %>% summarise(mu=mean(val)) %>% 
  dcast(gene~eth,value.var='mu') %>% mutate(diff=Caucasian-Asian)

# Make ggplot
gg.4genes <-
ggplot(long1,aes(x=ifelse(year>2003,'2004+',year),y=val,color=eth)) + 
  geom_jitter(size=2,show.legend = T) +
  theme(legend.position = 'bottom') +
  facet_wrap(~gene,scales = 'free') + labs(y='Gene expression') +
  theme(axis.title.x = element_blank()) +
  scale_color_discrete(name='Ethncitiy:')

gg_color_hue <- function(n) {
  hues = seq(15, 375, length = n + 1)
  hcl(h = hues, l = 65, c = 100)[1:n]
}

#### ---- VOLCANOS FOR CAUCASIAN 2002 vs 2003 ---- ####

white.idx <- GSE.meta %>% mutate(Year=format(Date,'%Y'),rid=1:nrow(.)) %>% 
  filter(eth2=='Caucasian' & Year %in% c('2002','2003')) %>%
  mutate(groups=ifelse(Year=='2002','white1','white2'))

caucT <- rowttests(x=GSE.exprs[,white.idx$rid],fac=factor(white.idx$groups)) %>% cbind(GSE.info,.) %>% 
  tbl_df %>% mutate(p.value=-log10(p.value)) %>% mutate(sidak=ifelse(p.value>sidak,T,F)) %>%
  filter( !(Probes %in% bad.probes) ) %>% 
  mutate(sig=ifelse(sidak & abs(dm)>0.5,T,F))

caucSig <- caucT %>% filter(sig) %>% mutate(dmp=sign(dm)) %>% count(dmp) %>%
  mutate(x=c(-1,1.5),y=c(13,13),label=paste(n))

c2 <- "#00BA38"

gg.volc2 <- 
  ggplot(caucT,aes(x=dm,y=p.value)) +
  geom_point(aes(color=sig),alpha=0.5,show.legend=F) +
  geom_hline(yintercept=sidak,color=c2,linetype=2) +
  geom_vline(xintercept = c(-0.5,0.5),color=c2,linetype=2) +
  labs(x='Mean difference',y='-log10(p-value)',subtitle='Caucasians 2002 vs 2003') +
  scale_color_manual(name='',values=c('grey',c2)) +
  geom_text(data=caucSig,inherit.aes=F,aes(x=x,y=y,label=label),color=c2,size=7) +
  geom_text(data=data.frame(x=1.5,y=sidak-1),inherit.aes = F,aes(x=x,y=y,label='Sidák'),color=c2,size=7)


#### ---- VOLCANOS FOR ASIAN 2004-6 vs CAUCASIAN 2004-6 ---- ####

asian.idx <- GSE.meta %>% mutate(Year=format(Date,'%Y'),rid=1:nrow(.)) %>% 
  filter(Year %in% c('2004','2005','2006'))

asianT <- rowttests(x=GSE.exprs[,asian.idx$rid],fac=asian.idx$eth2) %>% cbind(GSE.info,.) %>% 
  tbl_df %>% mutate(p.value=-log10(p.value)) %>% mutate(sidak=ifelse(p.value>sidak,T,F)) %>%
  filter( !(Probes %in% bad.probes) ) %>% 
  mutate(sig=ifelse(sidak & abs(dm)>0.5,T,F))

asianSig <- asianT %>% filter(sig) %>% mutate(dmp=sign(dm)) %>% count(dmp) %>%
  mutate(x=c(-1.5,2.5),y=c(15,15),label=paste(n))

c3 <- "#619CFF"

gg.volc3 <- 
  ggplot(asianT %>% filter(p.value<20),aes(x=dm,y=p.value)) +
  geom_point(aes(color=sig),alpha=0.5,show.legend=F) +
  geom_hline(yintercept=sidak,color=c3,linetype=2) +
  geom_vline(xintercept = c(-0.5,0.5),color=c3,linetype=2) +
  labs(x='Mean difference',y='-log10(p-value)',subtitle='Caucasians vs Asian 2004-6') +
  scale_color_manual(name='',values=c('grey',c3)) +
  geom_text(data=asianSig,inherit.aes=F,aes(x=x,y=y,label=label),color=c3,size=7) +
  geom_text(data=data.frame(x=3,y=sidak+2),inherit.aes = F,aes(x=x,y=y,label='Sidák'),color=c3,size=7)

#### ---- MC Simulations ---- ####

# Get Asian index
asn.idx <- which(GSE.meta$eth2=='Asian')
# Get 2002/3 caucasian index
widx.0203 <- GSE.meta %>% mutate(Year=format(Date,'%Y'),rid=1:nrow(.)) %>% filter(Year<=2003) %>% use_series(rid)

# Set up simulation
nsim <- 500
nwhite <- asian.idx %>% filter(eth2=='Caucasian') %>% nrow
nsig <- rep(NA,nsim)

# Loop
for(k in 1:nsim){
  set.seed(k)
  rand.col <- sample(widx.0203,nwhite,replace=F) # Randomly select 16 columns
  nsig[k] <- 
    rowttests(GSE.exprs[,c(rand.col,asn.idx)],factor(c(rep('White',nwhite),rep('Asian',length(asn.idx))))) %>% 
    tbl_df %>% mutate(plog10=-log10(p.value),sig=ifelse(plog10>sidak & abs(dm)>0.5,T,F)) %>%
    filter(sig) %>% nrow
}

# Get the number of significant
ndiff.0405 <- asianT %>% filter(sig) %>% nrow

# Now make a histogram chart and compare to the ndiff.0405
gg.mc <-
  ggplot(data.frame(nsig),aes(x=nsig,y=..count..)) + 
  geom_histogram(color='purple',fill='grey',bins=35) + 
  geom_vline(xintercept=ndiff.0405,color=c1,linetype=2) +
  geom_text(aes(x=ndiff.0405+100,y=50,label=ndiff.0405),color=c1) + 
  xlab('# of significant genes') + ylab('Frequency') + 
  scale_x_continuous(breaks=seq(0,1500,500),labels=seq(0,1500,500),limits=c(0,1500)) + 
  labs(subtitle='MC simulations of 16 Caucasian from 2002/03',caption='Significant implies p>Sidák and abs(val)>0.5')

#### ---- BATCH DRIVERS? ---- ####

# Get the minimum  Asian date
min.Date <- GSE.meta %>% filter(eth2=='Asian') %>% use_series(Date) %>% min

# Analyze the VRK3 gene
vrk3 <- tibble(Date=GSE.meta$Date,Eth3=GSE.meta$eth,Eth2=GSE.meta$eth2,
         Gene=GSE.exprs[which(GSE.info$Symbols=='VRK3'),]) %>%
          mutate(diffDate=(Date-min.Date) %>% as.numeric)

# Get the Caucasian average
vrk3.white <- vrk3 %>% mutate(Year=format(Date,'%Y') %>% ifelse(.>2003,'2004+',.)) %>% filter(Eth2=='Caucasian') %>% 
  group_by(Year) %>% summarise(mu=mean(Gene))
# Get the different batches
vrk3.asian.batches <- vrk3 %>% group_by(Date,Eth2) %>% 
  summarise(mu=mean(Gene),mi=min(Gene),ma=max(Gene)) %>% filter(Eth2=='Asian') %>%
  mutate(diffDate=(Date-min.Date) %>% as.numeric)
# Get the low and high text
odd <- which(mod(1:nrow(vrk3.asian.batches),2)==1)
even <- which(mod(1:nrow(vrk3.asian.batches),2)==0)
# Get breaks
vrk3.x <- vrk3.asian.batches[c(1,6,9,12),]

gg.vrk3 <-
ggplot(vrk3.asian.batches,aes(x=diffDate,y=mu)) + 
  geom_point(size=3) + geom_line() + 
  geom_jitter(data=vrk3 %>% filter(Eth2=='Asian'),aes(x=diffDate,y=Gene),inherit.aes = F,color='red',alpha=0.5) + 
  labs(y='VRK3 expression',subtitle='Asian expression by batch\nBlack dots show mean\nBlue line Caucasian average by date') + 
  theme(axis.title.x=element_blank(),axis.text.x=element_text(angle=60,hjust=1,size=10)) +
  scale_x_continuous(breaks=vrk3.x$diffDate,labels=format(vrk3.x$Date,'%d-%b-%y')) + 
  geom_hline(yintercept=vrk3.white$mu,color='blue',linetype=2) +
  geom_text(data=vrk3.white,inherit.aes=F,aes(x=260,y=mu,label=Year),color='blue',nudge_y=0.2,size=7)

###################################################
###### ----- STEP 3: EX-ANTE SPOTTING ----- #######
###################################################

# ---- Principal component analysis ---- #

# Get the SVD for the gene-normalized data
GSE.norm <- (GSE.exprs - rowMeans(GSE.exprs))/rowSds(GSE.exprs) 
GSE.svd  <- svd( GSE.norm )
# Get the share of variance explained by the factors
GSE.share <- with(GSE.svd,data.frame(nshare=d^2/sum(d^2)*100,npc=1:length(d))) %>% 
  tbl_df %>% mutate(cumshare=cumsum(nshare))
# Use first two PCs for correlation
GSE.12 <- GSE.svd$v[,1:2] %>% set_colnames(paste('PC',1:ncol(.),sep=''))
GSE.12.tbl <- with(GSE.meta,cbind(GSE.12,tibble(Ethnicity=eth2,Date=Date))) %>% 
  tbl_df %>% arrange(Date) %>% mutate(DateMon=format(Date,'%b-%y')) %>% 
  gather(PC,Value,-Ethnicity,-Date,-DateMon) %>% mutate(DateMon=factor(DateMon) %>% fct_reorder(Date))

# gg it
gg.pc <- ggplot(GSE.12.tbl,aes(y=Value,x=DateMon,color=fct_rev(DateMon))) + 
  geom_boxplot(show.legend=F) + facet_wrap(~PC,ncol=2) + coord_flip() + 
  theme(axis.text.y=element_text(size=8),axis.title.y=element_blank(),axis.title.x=element_blank()) +
  labs(subtitle='Box plot for first 2 PCs by year-month of sequence')
# gg.pc

# ---- Classical MDS ---- #

# Define matrix, and rotate. matrix(1:40,ncol=5
EX <- GSE.norm %>% t
n <- nrow(EX)

# Calculate D
D <- dist(EX,diag=T,upper=T)^2 %>% as.matrix
# Calculate J
J <- diag(n) - (1/n) * matrix(1,ncol=n,nrow=n)
# Calculate B 
B <- (-1/2)*(J %*% D %*% J)
# Get the two largest eigenvalues/vectors
E <- eigen(B)
Lambda.m <- diag(E$values[1:2])
E.m <- E$vectors[,1:2]
# Get X
X <- E.m %*% sqrt(Lambda.m)

# Now get the data 
mds.dat <- X %>% tbl_df %>% set_colnames(str_c('MDS',1:2)) %>% 
  mutate(eth=ethnicity2,date=GSE.meta$Date,datey=format(date,'%Y'))

gg.mds <- ggplot(mds.dat,aes(x=MDS1*-1,y=MDS2,color=eth,shape=datey)) + 
  geom_point(size=3) + guides(shape=FALSE) + 
  scale_shape_manual(name='Date',values=paste(2:6)) +
  scale_color_discrete(name='Ethnicity:') + 
  labs(x=expression(MDS[1]),y=expression(MDS[2]),subtitle='MDS scaling shape: sequence year') +
  theme(legend.position = 'right')

########################################################
###### ----- STEP 4: Adjusting with COMBAT ----- #######
########################################################

# Get the batch factor
batch <- format(GSE.meta$Date,'%b-%y') %>% factor
# Run ComBat
combat_edata = ComBat(dat=GSE.exprs,batch=batch,par.prior=TRUE, prior.plots=FALSE)
# Run t-ttests
combat.tt <- rowttests(combat_edata,ethnicity2) %>% cbind(GSE.info,.) %>% 
  tbl_df %>% mutate(p.value=-log10(p.value),sidak=ifelse(p.value>sidak,T,F),sig=ifelse(sidak & abs(dm)>0.5,T,F))

c4 <- "indianred"
gg.combat <- 
  ggplot(combat.tt,aes(x=dm,y=p.value)) +
  geom_point(aes(color=sig),alpha=0.5,show.legend=F) +
  geom_hline(yintercept=sidak,color=c4,linetype=2) +
  geom_vline(xintercept = c(-0.5,0.5),color=c4,linetype=2) +
  labs(x='Mean difference',y='-log10(p-value)',subtitle='ComBat adjusted data (Caucasian vs Asian)') +
  scale_color_manual(name='',values=c('grey',c4)) +
  geom_text(data=data.frame(x=0,y=sidak-0.5),inherit.aes = F,aes(x=x,y=y,label='Sidák'),color=c4,size=7)

# Robustness check by gender
chrY.idx <- which(GSE.info$Chromosome=='chrY')
chrXY.idx <- which(GSE.info$Chromosome %in% c('chrY','chrX'))
chrY.e <- GSE.exprs[chrY.idx,] %>% colMeans 
# Get our gender index
gender.idx <- ifelse(chrY.e>mean(chrY.e),'M','F') %>% factor

# Get the ComBat t-tests
gender.combat <- rowttests(combat_edata[chrXY.idx,],gender.idx) %>% cbind(GSE.info[chrXY.idx,],.) %>% 
  tbl_df %>% mutate(pv=-log10(p.value))
# Overall 
gender.unadj <- rowttests(GSE.exprs[chrXY.idx,],gender.idx) %>% cbind(GSE.info[chrXY.idx,],.) %>% 
  tbl_df %>% mutate(pv=-log10(p.value))
# Combine
gender.dat <- gender.combat %>% dplyr::select(Symbols,dm,pv) %>% filter(pv>sidak) %>% 
  left_join(gender.unadj %>% dplyr::select(Symbols,dm,pv),by='Symbols') %>% 
  gather(var,val,-Symbols) %>% separate(var,into=c('stat','type'),sep='[.]') %>%
  mutate(type=fct_recode(type,'pre'='y','post'='x')) %>% dcast(Symbols+type~stat,value.var='val') %>%
  mutate(sig=pv>sidak)

# Make a plot
gg.gender <- 
ggplot(gender.dat,aes(y=Symbols,x=dm)) + 
  geom_point(size=3,aes(fill=type,color=type,shape=sig)) + 
  scale_color_discrete(name='',labels=c('pre-ComBat','post-Combat')) + 
  scale_shape_manual(name='>Sidák',labels=c('No','Yes'),values=c(4,21)) +
  labs(x='Mean difference',subtitle='Men vs Women (sex chromosome genes)\nAll differentially expressed genes') +
  theme(axis.text.y=element_text(size=8),legend.position=c(0.85,0.6),axis.title.y=element_blank()) +
  guides(fill=F)

############################################
######## SAVE DATA FOR MARKDOWN ############
############################################

rmd.list <- 
  list(gg.toy=gg.toy,
       GSE.meta=GSE.meta,
       gg.seqYear=gg.seqYear,
       GSE.info=head(GSE.info,4),
       nr=nrow(GSE.info),
       gg.eth.scat=gg.eth.scat,
       spielSig=sum(spielSig$n),
       gg.volc1=gg.volc1,
       gg.4genes=gg.4genes,
       gg.volc2=gg.volc2,
       gg.volc3=gg.volc3,
       gg.mc=gg.mc,
       gg.vrk3=gg.vrk3,
       caucSig=sum(caucSig$n),
       asianSig=sum(asianSig$n),
       nsig=mean(nsig),
       c1=c1,c2=c2,c3=c3,ndiff.0405=ndiff.0405,
       gg.pc=gg.pc,
       gg.mds=gg.mds,
       gg.combat=gg.combat,c4=c4,
       gg.gender=gg.gender)

save(rmd.list,file='rmd_data_batch.RData')

# load('rmd_data_batch.RData')


# rmd.list[['gg.toy']] <- gg.toy



# attach(rmd.list)
