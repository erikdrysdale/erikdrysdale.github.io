#' This R script will process all R mardown files (those with in_ext file extention,
#' .rmd by default) in the current working directory. Files with a status of
#' 'processed' will be converted to markdown (with out_ext file extention, '.markdown'
#' by default). It will change the published parameter to 'true' and change the
#' status parameter to 'publish'.
#' 
#' @param path_site path to the local root storing the site files
#' @param dir_rmd directory containing R Markdown files (inputs)
#' @param dir_md directory containing markdown files (outputs)
#' @param url_images where to store/get images created from plots directory +"/" (relative to path_site)
#' @param out_ext the file extention to use for processed files.
#' @param in_ext the file extention of input files to process.
#' @param recursive should rmd files in subdirectories be processed.
#' @return nothing.
#' @author Jason Bryer <jason@bryer.org> edited by Andy South

# setwd("C:/Users/erikinwest/Documents/bioeconometrician/github/erikdrysdale.github.io/")
# path_site = getwd();dir_rmd = "_rmd";dir_md = "_posts"
# url_images = "figures/";out_ext='.md';in_ext='.rmd';recursive=FALSE

rm(list=ls())

rmd2md <- function( path_site = getwd(),
                    dir_rmd = "_rmd",
                    dir_md = "_posts",                              
                    #dir_images = "figures",
                    url_images = "figures/",
                    out_ext='.md', 
                    in_ext='.rmd', 
                    recursive=FALSE) {
  
  require(knitr, quietly=TRUE, warn.conflicts=FALSE)
  
  #andy change to avoid path problems when running without sh on windows 
  files <- list.files(path=file.path(path_site,dir_rmd), pattern=in_ext, ignore.case=TRUE, recursive=recursive)
  
  for(f in files) {
    message(paste("Processing ", f, sep=''),encoding = "UTF-8")
    content <- readLines(file.path(path_site,dir_rmd,f))
    # If any html calls, replace the src=figures/... with src=/figures/...
    src.idx <- grep('src=',content,value=F)
    if (length(src.idx)>0) {
    content[src.idx] <- gsub('src=\"figures','src=\"/figures',content[src.idx])
    } else {}
    frontMatter <- which(substr(content, 1, 3) == '---')
    if(length(frontMatter) >= 2 & 1 %in% frontMatter) {
      statusLine <- which(substr(content, 1, 7) == 'status:')
      publishedLine <- which(substr(content, 1, 10) == 'published:')
      if(statusLine > frontMatter[1] & statusLine < frontMatter[2]) {
        status <- unlist(strsplit(content[statusLine], ':'))[2]
        status <- sub('[[:space:]]+$', '', status)
        status <- sub('^[[:space:]]+', '', status)
        if(tolower(status) == 'process') {
          #This is a bit of a hack but if a line has zero length (i.e. a
          #black line), it will be removed in the resulting markdown file.
          #This will ensure that all line returns are retained.
          content[nchar(content) == 0] <- ' '
          message(paste('Processing ', f, sep=''))
          content[statusLine] <- 'status: publish'
          content[publishedLine] <- 'published: true'
          
          #andy change to path
          outFile <- file.path(path_site, dir_md, paste0(substr(f, 1, (nchar(f)-(nchar(in_ext)))), out_ext))
          
          #render_markdown(strict=TRUE)
          #render_markdown(strict=FALSE) #code didn't render properly on blog
          
          #andy change to render for jekyll
          render_jekyll(highlight = "pygments")
          #render_jekyll(highlight = "prettify") #for javascript
          
          opts_knit$set(out.format='markdown') 
          
          # andy BEWARE don't set base.dir!! it caused me problems
          # "base.dir is never used when composing the URL of the figures; it is 
          # only used to save the figures to a different directory. 
          # The URL of an image is always base.url + fig.path"
          # https://groups.google.com/forum/#!topic/knitr/18aXpOmsumQ
          
          # Get data directory
          opts_knit$set(root.dir = dir_rmd)
          
          opts_knit$set(base.url = "/")
          # opts_knit$set(fig.width = 10)
          
          opts_chunk$set(fig.path = url_images)
          # opts_chunk$set(fig.width = 10)
          
          #andy I could try to make figures bigger
          #but that might make not work so well on mobile
          opts_chunk$set(fig.width  = 8.5,
                        fig.height = 7.5,
                        dpi=300)
          
          try(knit(text=content, output=outFile,encoding = "UTF-8"), silent=FALSE)
          
        } else {
          warning(paste("Not processing ", f, ", status is '", status, 
                        "'. Set status to 'process' to convert.", sep=''))
        }
      } else {
        warning("Status not found in front matter.")
      }
    } else {
      warning("No front matter found. Will not process this file.")
    }
  }
  invisible()
}


setwd("C:/Users/erikinwest/Documents/bioeconometrician/github/erikdrysdale.github.io/")
rmd2md()


# c1 <- 'black'
# g1 <- ggplot(gather(iris,var,val,-Species) %>% tbl_df,aes(x=val,y=..density..)) + 
#   geom_density(aes(fill=Species),color=c1) + facet_wrap(~var)
# 
# # Save data in a list
# rmd.list <- list(g1=g1,c1=c1)
# save(rmd.list,file='C:/Users/erikinwest/Documents/bioeconometrician/github/erikdrysdale.github.io/_rmd/rmd_data_test.RData')


# t1 <- readLines( "C:/Users/erikinwest/Documents/bioeconometrician/github/erikdrysdale.github.io/_posts/2016-12-28-batch_effects.md")
# t2 <- readLines( "C:/Users/erikinwest/Documents/bioeconometrician/github/erikdrysdale.github.io/_posts/2016-12-28-old_batch_effects.md")
# 
# t1[which(!t1==t2)]
# t2[which(!t1==t2)]
