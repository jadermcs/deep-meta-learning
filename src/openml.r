library(OpenML)
library(foreign)
library(ECoL)

df = listOMLDataSets()

sel.ids = which(
  df$status == "active" &
  df$number.of.instances <= 10000 &
  df$number.of.features <= 500 &
  df$number.of.missing.values == 0 &
  df$number.of.classes >= 2 &
  df$number.of.classes <= 10
)

sub.df = df[sel.ids,]

generate <- function(i) {

  did = sub.df[i, "data.id"]
  dataset = OpenML::getOMLDataSet(data.id = did)
  name = sub.df[i, "name"]

  aux = dataset$data

  if(dataset$target.feature != "class") {
    aux$class = dataset$data[,dataset$target.features]
    aux[,dataset$target.features] = NULL
  }

  if(!is.na(dataset$desc$ignore.attribute)) {
    aux[,dataset$desc$ignore.attribute] = NULL
  }

  aux$class = as.factor(as.numeric(aux$class))

  colnames(aux) = make.names(colnames(aux))
  foreign::write.arff(x = aux, file = paste0(did, "_", name, ".arff"))
}

for(i in 1:nrow(sub.df)) {
  try(generate(i), silent=TRUE)
}

remove <- function(file) {

  data = foreign::read.arff(file)

  if(nrow(data) == 0) {
    system(paste("rm ", file))
    return(0)
  }

  if(ncol(data) <= 1) {
    system(paste("rm ", file))
    return(0)
  }

  if(min(summary(data$class)) < 10) {
    system(paste("rm ", file))
    return(0)
  }

  aux = sapply(1:ncol(data), function(i) {
    if(is.factor(data[,i]))
      if(nlevels(data[,i]) >= 53)
        return(1)
    return(0)
  })

  if(any(aux == 1)) {
    system(paste("rm ", file))
    return(0)
  }
}

files = list.files()

for(i in files) {
  print(i)
  try(remove(i), silent=TRUE)
}

extract <- function(file) {
  data = foreign::read.arff(file)
  c(ncol(data), nlevels(data$class), max(summary(data$class)/nrow(data)),
    min(summary(data$class)/nrow(data)), 
    balance(class ~., data, measures=c("C1","C2")))
}

files = list.files()

aux = sapply(files, function(file) {
  print(file)
  extract(file)
})

aux = t(aux)

aux1 = rownames(aux[duplicated(aux), ])

for(i in aux1)
  system(paste("rm ", i))

