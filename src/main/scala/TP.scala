import org.apache.spark.SparkContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{
  LogisticRegression,
  RandomForestClassifier
}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.{SQLContext, DataFrame, Row, Dataset}
import org.apache.spark.ml.param.{
  DoubleParam,
  IntParam,
  ParamMap
}
import org.apache.spark.ml.tuning.{
  TrainValidationSplitModel,
  TrainValidationSplit,
  ParamGridBuilder
}
import org.apache.spark.ml.feature.{
  CountVectorizer,
  CountVectorizerModel,
  IDF,
  StringIndexer,
  VectorAssembler,
  OneHotEncoderEstimator
}
import org.apache.spark.ml.feature.{
  RegexTokenizer,
  StopWordsRemover
}

object TP {
  def getIdfStage(): IDF = {
    return new IDF()
      .setInputCol("tf")
      .setOutputCol("tfidf")
  }
  def getTokenizeStage(): RegexTokenizer = {
    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")

    return tokenizer
  }
  def getWordsRemoverStage(): StopWordsRemover = {
    val remover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("filtered_tokens")

    return remover
  }
  def getDf(sqlContext: SQLContext): DataFrame = {
    return sqlContext.read.parquet(
      "prepared_trainingset")
  }
  def getTfStage(): CountVectorizer = {
    return new CountVectorizer()
      .setInputCol("filtered_tokens")
      .setOutputCol("tf")
  }
  def indexCountryStage(): StringIndexer = {
    return new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country2_indexed")
      .setHandleInvalid("skip")
  }
  def indexCurrencyStage(): StringIndexer = {
    return new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency2_indexed")
      .setHandleInvalid("skip")
  }
  def getCurrencyOneHot(): OneHotEncoderEstimator = {
    return new OneHotEncoderEstimator()
      .setInputCols(Array("currency2_indexed"))
      .setOutputCols(Array("currency_onehot"))
  }
  def getCountryOneHot(): OneHotEncoderEstimator = {
    return new OneHotEncoderEstimator()
      .setInputCols(Array("country2_indexed"))
      .setOutputCols(Array("country_onehot"))
  }
  def VectorizationStage(): VectorAssembler = {
    return new VectorAssembler()
      .setInputCols(
        Array(
          "tfidf",
          "days_campaign",
          "hours_prepa",
          "goal",
          "country_onehot",
          "currency_onehot"
        ))
      .setOutputCol("features")
  }
  def getLogisticRegressionStage(): LogisticRegression = {
    return new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(300)
  }
  def getRandomForestStage(): RandomForestClassifier = {
    return new RandomForestClassifier()
      .setLabelCol("final_status")
      .setFeaturesCol("features")
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
  }
  def getModel(pipeline: Pipeline, paramGrid: Array[ParamMap], training: Dataset[Row]): TrainValidationSplitModel = {
    return new TrainValidationSplit()
      .setEvaluator(
        // defaults to f1 scoring
        // https://spark.apache.org/docs/2.3.2/api/java/org/apache/spark/ml/evaluation/MulticlassClassificationEvaluator.html#metricName--
        new MulticlassClassificationEvaluator()
          .setLabelCol("final_status")
          .setPredictionCol("predictions")
      )
      .setEstimator(pipeline)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7)
      .fit(training)
  }
  def getParamGridRandomForest(numTrees: IntParam, minDf: DoubleParam): Array[ParamMap] = {
    return new ParamGridBuilder()
      .addGrid(numTrees, Array(10, 500, 10))
      .addGrid(minDf, Array(55.0, 75.0, 95.0))
      .build()
  }
  def getParamGridLogisticReg(regParam: DoubleParam, minDf: DoubleParam): Array[ParamMap] = {
    return new ParamGridBuilder()
      .addGrid(regParam, Array(10e-8, 10e-6, 10e-4, 10e-2))
      .addGrid(minDf, Array(55.0, 75.0, 95.0))
      .build()
  }
  def main(args: Array[String]) {
    val sc = SparkContext.getOrCreate()
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._

    val df = this.getDf(sqlContext)
    val stage1 = this.getTokenizeStage()
    val stage2 = this.getWordsRemoverStage()
    val stage3 = this.getTfStage()
    val stage4 = this.getIdfStage()
    val stage5 = this.indexCountryStage()
    val stage6 = this.indexCurrencyStage()
    val stage7 = this.getCurrencyOneHot()
    val stage8 = this.getCountryOneHot()
    val stage9 = this.VectorizationStage()
    val stage10LogisticReg = this.getLogisticRegressionStage()
    val stage10RandomForest = this.getRandomForestStage()

    val pipeline = new Pipeline()
      .setStages(
        Array(
          stage1,
          stage2,
          stage3,
          stage4,
          stage5,
          stage6,
          stage7,
          stage8,
          stage9,
          stage10LogisticReg
        ))

    val Array(training, test) = df.randomSplit(Array(0.9, 0.1), seed = 12345)

    // val paramGrid = this.getParamGridRandomForest(stage10RandomForest.numTrees, stage3.minDF)
    val paramGrid = this.getParamGridLogisticReg(stage10LogisticReg.regParam, stage3.minDF)
    val model = this.getModel(pipeline, paramGrid, training)
    val df_WithPredictions= model.transform(test)

    val score = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
      .evaluate(df_WithPredictions)

    df_WithPredictions.groupBy("final_status", "predictions").count.show()
    println("f1 score for model: " + score)
  }
}

// val args: Array[String] = new Array(0)
// 
// TP.main(args)
