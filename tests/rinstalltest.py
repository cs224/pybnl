
import rpy2, rpy2.rinterface, rpy2.robjects, rpy2.robjects.packages, rpy2.robjects.lib, rpy2.robjects.lib.grid, \
    rpy2.robjects.lib.ggplot2, rpy2.robjects.pandas2ri, rpy2.interactive.process_revents, \
    rpy2.interactive, rpy2.robjects.lib.grdevices

# rpy2.interactive.process_revents.start()
# rpy2.robjects.pandas2ri.activate()

# import R's "base" package
base = rpy2.robjects.packages.importr('base')
# import R's utility package
utils = rpy2.robjects.packages.importr('utils')
# select a mirror for R packages
utils.chooseCRANmirror(ind=1)  # select the first mirror in the list

rpy2.robjects.r('''
            library(curl)
            # library(RCurl)
            # options(download.file.method="libcurl", url.method="libcurl")
            # options(RCurlOptions = list(cainfo = system.file("CurlSSL", "cacert.pem", package = "RCurl"))) 
            # install.packages("devtools")
            # install.packages("printr")
            # install.packages("bnlearn")
            # print(Sys.getenv("http_proxy"))
            # print(Sys.getenv("https_proxy"))
            # print(Sys.getenv("HTTP_PROXY"))
            # print(Sys.getenv("HTTPS_PROXY"))
            
            # url = "https://bioconductor.org/biocLite.R"
            # download.file(url, tempfile())
            
            
            # biocLite("graph", suppressUpdates=TRUE)
            # biocLite("RBGL", suppressUpdates=TRUE)
            # install.packages("gRbase")
            # install.packages("gRain")

            tmp <- tempfile()
            curl_download("https://bioconductor.org/biocLite.R", tmp)
            source(tmp)
            biocLite(c("graph","RBGL"), suppressUpdates=TRUE, suppressAutoUpdate=TRUE)
            
            list.of.packages <- c("devtools", "printr", "bnlearn", "gRbase", "gRain")
            new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
            if(length(new.packages)) install.packages(new.packages)
    ''')

# R package names
packnames = ('bnlearn', 'gRain')

# R vector of strings

# Selectively install what needs to be install.
# We are fancy, just because we can.
names_to_install = [x for x in packnames if not rpy2.robjects.packages.isinstalled(x)]
if len(names_to_install) > 0:
    utils.install_packages(rpy2.robjects.StrVector(names_to_install))

grdevices = rpy2.robjects.packages.importr('grDevices')
bnlearn = rpy2.robjects.packages.importr('bnlearn')
gRain = rpy2.robjects.packages.importr('gRain')
