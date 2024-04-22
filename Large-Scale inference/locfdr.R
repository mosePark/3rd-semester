install.packages("locfdr")
library(locfdr)


data(hivdata)
w <- locfdr(hivdata)

summary(hivdata)
str(hivdata)

? locfdr

getAnywhere(locfdr)
