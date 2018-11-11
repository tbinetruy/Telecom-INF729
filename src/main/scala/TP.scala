import org.apache.spark.SparkContext
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.{SQLContext, DataFrame, Row, Dataset}
import org.apache.spark.ml.tuning.{TrainValidationSplitModel, TrainValidationSplit, ParamGridBuilder}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.param.{DoubleParam, ParamMap}
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

// TP3
object TP {
  def getIdfStage(): IDF = {
    return new IDF()
      .setInputCol("tf")
      .setOutputCol("tfidf")
  }
  def getTokenizeStage(): RegexTokenizer = {
    println("hello from foo")
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
      "../../../spark-tp3/prepared_trainingset")
  }
  def getTfStage(): CountVectorizer = {
    return new CountVectorizer()
      .setInputCol("filtered_tokens")
      .setOutputCol("tf")
      .setVocabSize(3)
      .setMinDF(2)
  }
  def indexCountryStage(): StringIndexer = {
    return new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country2_indexed")
  }
  def indexCurrencyStage(): StringIndexer = {
    return new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency2_indexed")
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
    new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      // .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("prediction")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(300)
  }
  def getModel(pipeline: Pipeline, paramGrid: Array[ParamMap], training: Dataset[Row]): TrainValidationSplitModel = {
    return new TrainValidationSplit()
      .setEvaluator(new MulticlassClassificationEvaluator)
      .setEstimator(pipeline)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7)
      .fit(training)
  }
  def getParamGrid(regParam: DoubleParam, minDf: DoubleParam): Array[ParamMap] = {
    return new ParamGridBuilder()
      .addGrid(regParam, Array(1.0E-8, 1.0E-2, 2.0))
      .addGrid(minDf, Array(55, 95, 20.0))
      .build()
  }
  def main(args: Array[String]) {
    val sc = SparkContext.getOrCreate()
    val yo: Array[String] = new Array(5)
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._

    val df = this.getDf(sqlContext)
    val df2 = df.withColumnRenamed("final_status", "label")
    val stage1 = this.getTokenizeStage()
    val stage2 = this.getWordsRemoverStage()
    val stage3 = this.getTfStage()
    val stage4 = this.getIdfStage()
    val stage5 = this.indexCountryStage()
    val stage6 = this.indexCurrencyStage()
    val stage7 = this.getCurrencyOneHot()
    val stage8 = this.getCountryOneHot()
    val stage9 = this.VectorizationStage()
    val stage10 = this.getLogisticRegressionStage()

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
          stage10
        ))

    val splits = df2.randomSplit(Array(0.1, 0.9), 24)
    val test = splits(0)
    val training = splits(1)

    val paramGrid = this.getParamGrid(stage10.regParam, stage3.minDF)
    val model = this.getModel(pipeline, paramGrid, training)
    val result = model.transform(test)


    println(df.head())
    println(result.head())
  }
}

// val args: Array[String] = new Array(0)
// 
// TP.main(args)
