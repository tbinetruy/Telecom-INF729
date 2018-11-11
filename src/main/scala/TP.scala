import org.apache.spark.SparkContext
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.{SQLContext, DataFrame}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
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
      .setInputCol("filtered_token")
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
          "currentcy_onehot"
        ))
      .setOutputCol("features")
  }
  def getLogisticRegressionStage(): LogisticRegression = {
    new LogisticRegression()
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
  def main(args: Array[String]) {
    val sc = SparkContext.getOrCreate()
    val yo: Array[String] = new Array(5)
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

    val splits = df.randomSplit(Array(0.1, 0.9), 24)
    val test = splits(0)
    val training = splits(1)


    println(df.head())
  }
}

// val args: Array[String] = new Array(0)
// 
// TP.main(args)
