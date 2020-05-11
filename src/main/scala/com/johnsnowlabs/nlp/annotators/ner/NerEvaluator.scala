package com.johnsnowlabs.nlp.annotators.ner
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

object NerEvaluator {

  def evaluateNER(spark: SparkSession = ResourceHelper.spark, ground_truth: Seq[String], predictions: Seq[String], percent:Boolean=true, mode: String = "entity_level"):  (Map[String, Float], DataFrame) ={
    //get seq of ground truth, seq of prediction
    //returns Map with avg precision, recall, f1
    //returns df with entity, precision, recall, f1, support
    //if mode==entity_level, remove O and I- and B- tags

    val entitiesNeedMetrics = if (mode == "entity_level") ground_truth.map(ent => ent.split("-").last) else ground_truth
    val predictionsCleaned = if (mode == "entity_level") predictions.map(ent => ent.split("-").last) else predictions
    val pairsCleaned = entitiesNeedMetrics zip predictionsCleaned
    var tpTotal = 0
    var fpTotal = 0
    var fnTotal = 0
    val distinctEntities = if (mode == "entity_level") (entitiesNeedMetrics ++ predictionsCleaned).distinct.filterNot(ent => ent == "O") else (entitiesNeedMetrics ++ predictionsCleaned).distinct
    val entityMetrics = distinctEntities.map(ent => {
      val tp = pairsCleaned.filter(pair => pair == (ent, ent)).length
      val fp = pairsCleaned.filter(pair => pair._1 != ent & pair._2 == ent).length
      val fn = pairsCleaned.filter(pair => pair._1 == ent & pair._2 != ent). length
      val support = pairsCleaned.filter(pair => pair._1 == ent).length

      var precision = if (tp + fp > 0) tp.toFloat/(tp+fp) else 0.toFloat
      var recall = if (tp + fn > 0 ) tp.toFloat/(tp+fn) else 0.toFloat
      var f1 = if (precision + recall > 0) 2.toFloat * precision * recall / (precision + recall) else 0.toFloat

      if (percent == true) {
        precision *= 100
        recall *= 100
        f1 *= 100
      }

      tpTotal += tp
      fpTotal += fp
      fnTotal += fn

      Row(ent, precision, recall, f1, support)
    })
    val schema = List(StructField("entity", StringType), StructField("precision", FloatType), StructField("recall", FloatType), StructField("f1", FloatType), StructField("support", IntegerType))

    var totalPrecision = if (tpTotal + fpTotal > 0) tpTotal.toFloat/(tpTotal+fpTotal) else 0.toFloat
    var totalRecall = if (tpTotal + fnTotal > 0 ) tpTotal.toFloat/(tpTotal+fnTotal) else 0.toFloat
    var totalF1 = if (totalPrecision + totalRecall > 0) 2 * totalPrecision * totalRecall / (totalPrecision + totalRecall) else 0.toFloat
    if (percent == true) {
      totalPrecision *= 100
      totalRecall *= 100
      totalF1 *= 100
    }



    val entityTotalMap = Map("total precision"->totalPrecision, "total recall"->totalRecall, "total f1"->totalF1)
    val entityLvlDf = spark.createDataFrame(spark.sparkContext.parallelize(entityMetrics), StructType(schema))
    (entityTotalMap, entityLvlDf)
  }


}
