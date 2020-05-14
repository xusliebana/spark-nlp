package com.johnsnowlabs.nlp.annotators.ner
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

object NerEvaluator {
  //TODO:
  //TODO: py4j
  //TODO: extend internal eval module

  def evaluateNer(spark: SparkSession = ResourceHelper.spark, ground_truth: Seq[String], predictions: Seq[String], percent:Boolean=true, outputDf:Boolean=false, mode: String = "entity_level"):
  (Map[String, (Any, Any, Float)], Map[String, (Float, Float, Float, Int)], Option[DataFrame])={
    //get seq of ground truth, seq of prediction
    //creates Map with avg precision, recall, f1
    //creates Map with entity, precision, recall, f1, support
    //if outputDf==true, creates df with entity, precision, recall, f1, support
    //returns a tuple of (Map, Map, option[df])
    //if mode==entity_level, remove O and I- and B- tags (for example, I-Location is the same label as B-Location in entity_level)
    //if percent==true, uses percents instead of decimals i

    val entitiesNeedMetrics = if (mode == "entity_level") ground_truth.map(ent => ent.split("-").last) else ground_truth //seq of entities that will need metrics
    val predictionsCleaned = if (mode == "entity_level") predictions.map(ent => ent.split("-").last) else predictions
    val pairsCleaned = entitiesNeedMetrics zip predictionsCleaned
    var tpTotal = 0 //total true positives for all entities
    var fpTotal = 0 //total false positives for all entities
    var fnTotal = 0 //total false negatives for all entities
    var supportTotal = 0 //total support for all entities
    val distinctEntities = if (mode == "entity_level") (entitiesNeedMetrics ++ predictionsCleaned).distinct.filterNot(ent => ent == "O") else (entitiesNeedMetrics ++ predictionsCleaned).distinct
    val entityMetrics = distinctEntities.map(ent => {
      val tp = pairsCleaned.filter(pair => pair == (ent, ent)).length //true positves for this entity label
      val fp = pairsCleaned.filter(pair => pair._1 != ent & pair._2 == ent).length//false positives for this entity label
      val fn = pairsCleaned.filter(pair => pair._1 == ent & pair._2 != ent). length//false negatives for this entity label
      val support = pairsCleaned.filter(pair => pair._1 == ent).length //support for this entity label

      var precision = if (tp + fp > 0) tp.toFloat/(tp+fp) else 0.toFloat //precision for this entity label
      var recall = if (tp + fn > 0 ) tp.toFloat/(tp+fn) else 0.toFloat //recall for this entity label
      var f1 = if (precision + recall > 0) 2.toFloat * precision * recall / (precision + recall) else 0.toFloat //f1 for this entity label

      if (percent == true) {
        precision *= 100
        recall *= 100
        f1 *= 100
      }

      tpTotal += tp
      fpTotal += fp
      fnTotal += fn
      supportTotal += support

      (ent, precision, recall, f1, support)
    })

    //macro values for precision, recall, f1
    val macroVals = entityMetrics.foldLeft((0.toFloat, 0.toFloat, 0.toFloat))( (a, b)=>
      (a._1 + b._2/entityMetrics.length, a._2 + b._3/entityMetrics.length, a._3 + b._3/entityMetrics.length) )

    //weighted values for precision, recall, f1
    val weightedVals = entityMetrics.foldLeft((0.toFloat, 0.toFloat, 0.toFloat)) ((a, b)=>{
      val weight = b._5.toFloat / supportTotal
      (a._1 + b._2*weight, a._2 + b._3*weight, a._3 + b._4*weight)})

    //overall accuracy
    val accuracy = if (percent) 100*tpTotal.toFloat/supportTotal else tpTotal.toFloat/supportTotal

    var microPrecision = if (tpTotal + fpTotal > 0) tpTotal.toFloat/(tpTotal+fpTotal) else 0.toFloat
    var microRecall = if (tpTotal + fnTotal > 0 ) tpTotal.toFloat/(tpTotal+fnTotal) else 0.toFloat
    var microF1 = if (microPrecision + microRecall > 0) 2 * microPrecision * microRecall / (microPrecision + microRecall) else 0.toFloat
    if (percent == true) {
      microPrecision *= 100
      microRecall *= 100
      microF1 *= 100
    }

    //overall metrics
    val averagesMap = Map("accuracy" -> (None, None, accuracy),
      "micro" -> (microPrecision, microRecall, microF1),
      "macro" -> macroVals,
      "weighted" -> weightedVals)

    //entity metrics stored as map
    val entitiesMap = entityMetrics.map(tup => (tup._1 -> (tup._2, tup._3, tup._4, tup._5))).
      foldLeft(Map.empty[String, (Float, Float, Float, Int)]) { case (m, (k, v)) => m.updated(k, v) }

    //entity metrics stored as df (only if outputDf is true)
    val entityMetricsDf = if (!outputDf) None else {
      val schema = List(StructField("entity", StringType), StructField("precision", FloatType), StructField("recall", FloatType), StructField("f1", FloatType), StructField("support", IntegerType))
      val entityMetricsRows=entityMetrics.map(tup => Row(tup._1, tup._2, tup._3, tup._4, tup._5))
      Some(spark.createDataFrame(spark.sparkContext.parallelize(entityMetricsRows), StructType(schema)))
    }

    (averagesMap, entitiesMap, entityMetricsDf)
  }



}
