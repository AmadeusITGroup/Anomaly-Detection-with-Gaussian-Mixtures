package com.amadeus.gmm_scoring

import org.apache.spark.ml.clustering.GaussianMixtureModel
import org.apache.spark.sql.functions._
import org.apache.spark.sql._

/*
 * This object collects funtions that allow to use a GaussianMixtureModel in anomaly detection
 *
 * Normally you will use a Pipeline to obtain a PipelineModel, one stage of which is a GaussianMixtureModel
 * Once the model is trained, you should have the GaussianMixtureModel and the Vector of features used in training
 *
 * Call the scoring function like this:
 *
 * val raw_score_column_name = "" // the column where you want to store the scores, defaults to "raw_score"
 * val normalized_score_column_name = "" // the column where you want to store the scores, defaults to "score"
 * val input_df : DataFrame = // your input dataset 
 * val gmm: GaussianMixtureModel = // get the trained GaussianMixtureModel 
 * val with_raw_score : DataFrame = input_df.withColumn(<raw_score_column_name>, GaussianMixtureModelWithScoring.scoreWith(gmm)(col("<the column holding features used in training>")))
 * val with_normalized_score : DataFrame = GaussianMixtureModelWithScoring.normalize_score(with_raw_score)
 *
 * Your input_df will now have a column named <normalized_score_column_name> holding the score for each input
 */
object GaussianMixtureModelWithScoring {

    /** User Defined Function that scores samples with the given GMM model
   *
   * @param gmm the GaussianMixtureModel
   * @param features a Spark ML Vector holding the features to use
   */
  def scoreWith(gmm: GaussianMixtureModel) = udf((features: org.apache.spark.ml.linalg.Vector) => score(gmm, features))


  /** Scores one sample with the given GMM model
   * 
   * @param gmm the GaussianMixtureModel
   * @param features a Spark ML Vector holding the features to use
   * @return a score for the given sample
   */
  def score(gmm: GaussianMixtureModel, features: org.apache.spark.ml.linalg.Vector): Double = {
    var probs = 0.0
    for (i <- 0 until gmm.getK) {
      val gaussian_i = gmm.gaussians(i)
      val probs_i = gmm.weights(i) * gaussian_i.pdf(features)
      probs = probs + probs_i
    }
    probs = probs +  Double.MinPositiveValue //To avoid being 0. This will be mac in csae probs is zero

    -math.log(probs)
  }  

  /** Normalizes scores of GMM across a Dataset to be in the range [0,1].
   * 
   * @param df the input DataFrame, which must include a column with raw scores computed with GMM
   * @param raw_score_column_name the name of the column holding raw scores. Defaults to "raw_score"
   * @param normalized_score_column_name the name of the column that will hold the normalized scores. Defaults to "normalized_score"
   * @return a DataFrame with an extra column holding normalized scores
   */
  def normalizeScore(df: DataFrame, raw_score_column_name : String = "raw_score", normalized_score_column_name : String = "normalized_score") : DataFrame = {
    //val (vMin, vMax) = ( -math.log(Double.MaxValue), -math.log(Double.MinPositiveValue) )
    val (vMin, vMax) = ( -math.log(Double.MaxValue), -math.log(Double.MinPositiveValue) )
    val scaledRange = lit(1) // Range of the scaled variable
    val scaledMin = lit(0)  // Min value of the scaled variable
    val vNormalized = (col(raw_score_column_name) - vMin) / (vMax - vMin) // score normalized to (0, 1) range
    val vScaled = scaledRange * vNormalized + scaledMin // This is only for reference, as it is useful only if the min is not 0

    df.withColumn(normalized_score_column_name, vScaled)
  }
}