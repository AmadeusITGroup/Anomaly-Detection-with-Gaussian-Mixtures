package com.amadeus.gmm_scoring

import org.scalatest._
import org.scalatest.Matchers._
import org.apache.spark.ml.clustering.{GaussianMixture, GaussianMixtureModel}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.SparkSession
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._


case class Sample(x: Double, y: Double)


object SparkSessionSpec {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val sparkSession = SparkSession.builder()
      .master("local")
      .appName("ML Test")
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      //.config("spark.kryo.registrator", "common.CustomRegistrator")  // for Joda classes if needed
      .getOrCreate()

    val sc = sparkSession.sparkContext
}


object GmmUtils {
  
  def getAssembler(inputCols: Array[String]) : VectorAssembler = {
    //val inputCols = Array("x", "y")
    new VectorAssembler().setInputCols(inputCols).setOutputCol("features")
  }

  def getGmm(k: Int) : GaussianMixture = new GaussianMixture().setFeaturesCol("features").setK(k)

  def getPipeline(inputCols: Array[String], k: Int) : Pipeline = {
    new Pipeline()
      .setStages(Array(
        getAssembler(inputCols),
        getGmm(k)
    ))
  }

  def train(input: DataFrame, inputCols: Array[String], k: Int) : PipelineModel = getPipeline(inputCols, k).fit(input)

  def getGmmModel(pipelineModel: PipelineModel) : GaussianMixtureModel = pipelineModel.stages(1).asInstanceOf[org.apache.spark.ml.clustering.GaussianMixtureModel]

  def predict(input: DataFrame, pipeline: PipelineModel) : DataFrame = pipeline.transform(input)


}


class GmmWithScoringSuite extends FlatSpec {

  "scoring" should "score 2 distinct 2D points" in {
    val sparkSession = SparkSessionSpec.sparkSession
    import sparkSession.implicits._

    val trainingDataset = Seq(
        Sample(0,0),
        Sample(2,0),
        Sample(0,2),
        Sample(4,0),
        Sample(0,4),
        Sample(4,4),
        Sample(2,2),
        Sample(4,2),
        Sample(2,4),
        Sample(1,1),
        Sample(3,3),
        Sample(1,4),                
        Sample(100,100),
        Sample(100,120),
        Sample(120,100),
        Sample(120,120),
        Sample(100,140),
        Sample(140,100),
        Sample(140,140),
        Sample(140,120),
        Sample(120,140),
        Sample(130,130),
        Sample(110,110),
        Sample(110,140)                
      ).toDF

    val testDataset = Seq(
      Sample(1,1),
      Sample(500,500)
    ).toDF

    val pipeline : PipelineModel = GmmUtils.train(trainingDataset, Array("x", "y"), 2)

    val predictions = GmmUtils.predict(testDataset, pipeline) 
    val gmm : GaussianMixtureModel = GmmUtils.getGmmModel(pipeline)

    val with_raw_score : DataFrame = predictions.withColumn("raw_score", GaussianMixtureModelWithScoring.scoreWith(gmm)(col("features")))

    /*
    +-----+-----+-------------+----------+-------------------------------------------+------------------+
    |x    |y    |features     |prediction|probability                                |raw_score         |
    +-----+-----+-------------+----------+-------------------------------------------+------------------+
    |1.0  |1.0  |[1.0,1.0]    |1         |[1.0407565019590488E-14,0.9999999999999896]|3.8474099416523395|
    |500.0|500.0|[500.0,500.0]|0         |[0.5,0.5]                                  |617.1895870128861 |
    +-----+-----+-------------+----------+-------------------------------------------+------------------+
    */

    val with_normalized_score : DataFrame = GaussianMixtureModelWithScoring.normalizeScore(with_raw_score)
    with_normalized_score.show(false)

    /*
    +-----+-----+-------------+----------+-------------------------------------------+------------------+--------------------+
    |x    |y    |features     |prediction|probability                                |raw_score         |normalized_score    |
    +-----+-----+-------------+----------+-------------------------------------------+------------------+--------------------+
    |1.0  |1.0  |[1.0,1.0]    |1         |[1.0407565019590488E-14,0.9999999999999896]|3.8474099416523395|0.005168192963769764|
    |500.0|500.0|[500.0,500.0]|0         |[0.5,0.5]                                  |617.1895870128861 |0.8290655088191791  |
    +-----+-----+-------------+----------+-------------------------------------------+------------------+--------------------+
    */

    // check that point in cluster has a relaitvely low score
    (with_normalized_score.filter(col("x") === 1.0).select("normalized_score").head.getDouble(0) < 0.5) shouldBe true

    // check that point away from the cluster has a high score
    (with_normalized_score.filter(col("x") === 500.0).select("normalized_score").head.getDouble(0) > 0.8) shouldBe true
  }


 it should "score 3 distinct 2D points" in {
    val sparkSession = SparkSessionSpec.sparkSession
    import sparkSession.implicits._

    val trainingDataset = Seq(
        Sample(0,0),
        Sample(2,0),
        Sample(0,2),
        Sample(4,0),
        Sample(0,4),
        Sample(4,4),
        Sample(2,2),
        Sample(4,2),
        Sample(2,4),
        Sample(1,1),
        Sample(3,3),
        Sample(1,4)          
      ).toDF

    val testDataset = Seq(
      Sample(1,1),
      Sample(250,250),
      Sample(500,500)
    ).toDF


    val pipeline : PipelineModel = GmmUtils.train(trainingDataset, Array("x", "y"), 2)

    val predictions = GmmUtils.predict(testDataset, pipeline) 
    val gmm : GaussianMixtureModel = GmmUtils.getGmmModel(pipeline)

    val with_raw_score : DataFrame = predictions.withColumn("raw_score", GaussianMixtureModelWithScoring.scoreWith(gmm)(col("features")))

    /*
    +-----+-----+-------------+----------+------------------------------------------+------------------+
    |x    |y    |features     |prediction|probability                               |raw_score         |
    +-----+-----+-------------+----------+------------------------------------------+------------------+
    |1.0  |1.0  |[1.0,1.0]    |0         |[0.9999999999999959,4.062676251191043E-15]|2.9067189258298014|
    |250.0|250.0|[250.0,250.0]|0         |[0.5,0.5]                                 |744.4400719213812 |
    |500.0|500.0|[500.0,500.0]|0         |[0.5,0.5]                                 |744.4400719213812 |
    +-----+-----+-------------+----------+------------------------------------------+------------------+
    */

    val with_normalized_score : DataFrame = GaussianMixtureModelWithScoring.normalizeScore(with_raw_score)
    with_normalized_score.show(false)

    /*
    +-----+-----+-------------+----------+------------------------------------------+------------------+--------------------+
    |x    |y    |features     |prediction|probability                               |raw_score         |normalized_score    |
    +-----+-----+-------------+----------+------------------------------------------+------------------+--------------------+
    |1.0  |1.0  |[1.0,1.0]    |0         |[0.9999999999999959,4.062676251191043E-15]|2.9067189258298014|0.003904570744462523|
    |250.0|250.0|[250.0,250.0]|0         |[0.5,0.5]                                 |744.4400719213812 |1.0                 |
    |500.0|500.0|[500.0,500.0]|0         |[0.5,0.5]                                 |744.4400719213812 |1.0                 |
    +-----+-----+-------------+----------+------------------------------------------+------------------+--------------------+
    */

    // check that point in cluster has a relatively low score
    (with_normalized_score.filter(col("x") === 1.0).select("normalized_score").head.getDouble(0) < 0.5) shouldBe true

    // check that point away from the cluster has a high score
    (with_normalized_score.filter(col("x") === 250.0).select("normalized_score").head.getDouble(0) > 0.8) shouldBe true
    (with_normalized_score.filter(col("x") === 500.0).select("normalized_score").head.getDouble(0) > 0.8) shouldBe true
  }  

}