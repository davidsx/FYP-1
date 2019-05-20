#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler


# In[2]:


crime_schema = StructType([StructField("IncidentName",IntegerType(),True),
                          StructField("Category",StringType(),True),
                          StructField("Descript",StringType(),True),
                          StructField("DayofWeek",StringType(),True),
                          StructField("Date",StringType(),True),
                          StructField("Time",StringType(),True),
                          StructField("PdDistrict",StringType(),True),
                          StructField("Resolution",StringType(),True),
                          StructField("Address",StringType(),True),
                          StructField("X",DoubleType(),True),
                          StructField("Y",DoubleType(),True),
                          StructField("Location",StringType(),True),
                          StructField("PdID",StringType(),True)])


# In[3]:


# Load and parse the data file, converting it to a DataFrame.
crimeDF = spark.read.csv('s3a://crimedatafyp/crimeData/Sample9.csv',header=True,schema=crime_schema)
crimeDF.count()


# # Data Cleaning

# # Data Transformation

# In[4]:


crimeDF.printSchema()


# In[5]:


dropList = ["IncidentName","Descript","Resolution","Address","Location","PdID","X","Y"]
crimeDF = crimeDF.select([column for column in crimeDF.columns if column not in dropList])
crimeDF.printSchema()


# In[6]:


from pyspark.sql.functions import *
crimeNewDF = crimeDF.withColumn('Year',year(unix_timestamp('Date', 'MM/dd/yyyy').cast("timestamp")))                    .withColumn('Month',month(unix_timestamp('Date', 'MM/dd/yyyy').cast("timestamp")))                    .withColumn('Day',dayofmonth(unix_timestamp('Date', 'MM/dd/yyyy').cast("timestamp")))                    .withColumn('Hour',hour('Time'))
crimeNewDF.show()


# In[7]:


crimeNewDF=crimeNewDF.drop('Date').drop('Time').drop('X').drop('Y')
crimeNewDF.show()


# In[8]:


cat_cols = [item[0] for item in crimeNewDF.dtypes if item[1].startswith('string')] 
print(str(len(cat_cols)) + '  categorical features')
num_cols = [item[0] for item in crimeNewDF.dtypes if item[1].startswith('int') | item[1].startswith('double')][1:]
print(str(len(num_cols)) + '  numerical features')


# In[9]:


#Index crime category
from pyspark.ml.feature import StringIndexer,IndexToString
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import OneHotEncoderEstimator
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

(trainingData,testData) = crimeNewDF.randomSplit([0.7,0.3],seed= 100)

catIndexer = StringIndexer(inputCol="Category",outputCol="label",handleInvalid='keep')

indexers = [StringIndexer(inputCol= column, outputCol=column+"_index") for column in list(set(crimeNewDF.columns)-set(['Year','Month','Day','Hour','Category']))]

encoders = [OneHotEncoderEstimator(
    inputCols=[indexer.getOutputCol()],
    outputCols=[indexer.getOutputCol()+"_encoded"]) for indexer in indexers]

assemblerInputs = [column + "_index_encoded" for column in list(set(crimeNewDF.columns)-set(['Year','Month','Day','Hour','Category']))] + num_cols

assembler = VectorAssembler(
            inputCols=assemblerInputs,
            outputCol="features")

#create the trainer
nb = NaiveBayes(smoothing=3.0,modelType="multinomial")

pipeline = Pipeline(stages=indexers+encoders+[catIndexer,assembler,nb])
modelDF = pipeline.fit(trainingData)
pr = modelDF.transform(testData)

        
#nb_model = nb.fit(trainingData)
#predictions = nb_model.transform(testData)
#predictions.select("Category","label","probability","prediction").show(n=10)


# In[10]:


evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
evaluator.evaluate(pr)


# In[11]:


pr.printSchema()


# In[15]:


pr.select("prediction","probability","label","Category").show()


# In[ ]:




