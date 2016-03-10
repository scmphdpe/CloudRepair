from pyspark import SparkContext, SparkConf
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel
import os
import matplotlib.pyplot as plt

##### cluster initialization
appName = "CloudRepairDEV"
master = local[*]
conf = SparkConf().setAppName(appName).setMaster(master)
sc = SparkContext(conf=conf)
######### cluster initialization

##### define functions
def rmse(pairRDD,n):
    return ((pairRDD.map(lambda x: (x[0]-x[1])**2).
                     reduce(lambda a,b:a+b))/float(n))**(0.5)

def parsePixels(line):
    try:
        values = [float(i) for i in line.split(',')]
        return LabeledPoint(values[0], # ndvi = label
                            [values[1]/1000000.0, # x coordinate
                            values[2]/1000000.0, # y coordinate
                            values[3], # month encoded to number# precipitation in inches
                            #values[4], # year
                            values[5], # rainfall in inches
                            values[6], # temperature
                            values[7]]) # mean tide gage value in meters
    except ValueError:
        # flag lines that do not conform
        return LabeledPoint(9., [0.,0.,0.,0.,0.])
######### define functions

##### comment out unless on databricks
# import urllib
# ACCESS_KEY = "AKIAJ6OBLAVGH3YAP6SA"
# SECRET_KEY = "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"
# ENCODED_SECRET_KEY = urllib.quote(SECRET_KEY, "")
# AWS_BUCKET_NAME = "databricksucf"
# MOUNT_NAME = "dataCR"
# dbutils.fs.mount("s3n://%s:%s@%s" % (ACCESS_KEY, ENCODED_SECRET_KEY, AWS_BUCKET_NAME), "/mnt/%s" % MOUNT_NAME)
# display(dbutils.fs.ls("/mnt/dataCR"))
# rawRDD = sc.textFile('/mnt/dataCR/')
######### databricks

##### stokes
rawRDD = sc.textFile('~/CloudRepair/DATA/input_clear.txt')
######### stokes

# make labeled points on subsample of data
sampleFraction = 0.05
opsRDD = rawRDD.sample(False,sampleFraction,59).map(lambda x: parsePixels(x)).filter(lambda x: x.label < 1.0)

##### MACHINE LEARNING SECTION
# create file for results
fOut = open('~/CloudRepair/DATA/results_2016-03-10.txt','w')

##### randomly divide data into training and test sets
training, test = opsRDD.randomSplit([0.7,0.3],seed=59)

# compute mean ndvi for comparison
meanNDVI = opsRDD.map(lambda x: x.label).mean()

##### use the MEAN NDVI as the prediction (for comparison)
meanRES = test.map(lambda x: (x.label,meanNDVI))
rmseVsMean = rmse(meanRES,numTest)
outString = "Simple Mean NDVI RMSE = " + str(rmseVsMean) + "\n\n"
fOut.write(outString)
######### MEAN NDVI

##### LINEAR REGRESSION WITH STOCHASTIC GRADIENT DESCENT
# if a model has already been trained, use it
# otherwise train a new one and save it
if os.path.exists('~/CloudRepair/MODELS/lrmCR'):
    lrm = LinearRegressionModel.load(sc,'~/CloudRepair/MODELS/lrmCR')
else:
    lrm = LinearRegressionWithSGD.train(training,
                                        iterations=10000,
                                        step=0.0000001,
                                        miniBatchFraction=0.10)
    lrm.save(sc, '~/CloudRepair/MODELS/lrmCR')

lrmPred = lrm.predict(test.map(lambda x: x.features))
lrmRES = test.map(lambda x: x.label).zip(lrmPred)
rmseLRM = rmse(lrmRES,numTest)
outString = "Linear Regression NDVI RMSE = " + str(rmseLRM) + "\n\n"
fOut.write(outString)
######### LINEAR REGRESSION WITH STOCHASTIC GRADIENT DESCENT

##### RANDOM FOREST optimization
rfd = []
rfn = []
rfRmse = []
for d in range(3,30):
    for n in range(11,401,10):
        rfModel = RandomForest.trainRegressor(training,
                                              categoricalFeaturesInfo={},
                                              numTrees=n,
                                              featureSubsetStrategy="auto",
                                              impurity='variance',
                                              maxDepth=d,
                                              maxBins=32)
        rfPred = rfModel.predict(test.map(lambda x: x.features))
        rfRES = test.map(lambda lp: lp.label).zip(rfPred)
        rmseRF = rmse(rfRES,numTest)
        outString = "RF NDVI RMSE " + str(d) + "/" + str(n) + " = " + str(rmseVsMean) + "\n"
        fOut.write(outString)
        rfd.append(d)
        rfn.append(n)
        rfRmse.append(rmseRF)

plt.scatter(rfn,rfd,s=rfRmse)
plt.savefig('~/CloudRepair/DATA/rfOut.png',dpi=300,bbox_inches='tight')

fOut.close()
