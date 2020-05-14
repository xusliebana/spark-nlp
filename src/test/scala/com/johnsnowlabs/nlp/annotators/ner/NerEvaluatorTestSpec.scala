package com.johnsnowlabs.nlp.annotators.ner

import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.apache.spark.sql.{DataFrame, Row}
import org.scalatest.FlatSpec

class NerEvaluatorTestSpec extends FlatSpec{


  val ground_truth = Seq("B-Location", "B-Age", "B-TestResult", "B-UnspecificTherapy", "B-Age", "O", "B-CancerDx", "O", "O", "O", "B-UnspecificTherapy", "B-UnspecificTherapy", "I-HormonalTherapy", "B-Age", "I-PlanHeader", "I-HormonalTherapy", "B-Location", "B-Location", "I-HormonalTherapy", "B-Age", "B-UnspecificTherapy", "O", "B-Age", "B-CancerDx", "O", "I-PlanHeader", "B-Location", "B-Age", "I-PlanHeader", "B-TestResult", "B-Age", "I-HormonalTherapy", "B-Age", "B-TestResult", "B-UnspecificTherapy", "B-Location", "O", "B-CancerDx", "B-TestResult", "B-CancerDx", "I-PlanHeader", "B-Age", "B-Age", "O", "I-HormonalTherapy", "B-UnspecificTherapy", "B-CancerDx", "B-TestResult", "O", "B-TestResult", "B-CancerDx", "B-Location", "B-Age", "O", "B-TestResult", "B-Age", "I-HormonalTherapy", "B-Location", "B-UnspecificTherapy", "B-CancerDx", "B-UnspecificTherapy", "B-TestResult", "I-HormonalTherapy", "I-PlanHeader", "I-PlanHeader", "B-CancerDx", "I-HormonalTherapy", "I-HormonalTherapy", "B-Location", "I-HormonalTherapy", "B-TestResult", "I-PlanHeader", "B-CancerDx", "I-PlanHeader", "B-Age", "I-PlanHeader", "B-Location", "B-Location", "B-CancerDx", "B-Location", "B-Location", "B-Age", "B-CancerDx", "B-CancerDx", "B-CancerDx", "B-TestResult", "I-HormonalTherapy", "B-TestResult", "B-Location", "I-HormonalTherapy", "B-Age", "O", "B-Location", "B-TestResult", "B-CancerDx", "B-CancerDx", "B-Location", "I-PlanHeader", "I-PlanHeader", "B-Age")
  val predictions = Seq("I-HormonalTherapy", "B-TestResult", "B-TestResult", "B-Location", "B-Location", "B-CancerDx", "B-CancerDx", "I-HormonalTherapy", "I-HormonalTherapy", "B-Age", "B-Location", "O", "B-TestResult", "I-PlanHeader", "B-Location", "B-TestResult", "B-Location", "B-TestResult", "B-CancerDx", "B-Location", "B-Location", "B-CancerDx", "I-PlanHeader", "B-UnspecificTherapy", "B-UnspecificTherapy", "I-HormonalTherapy", "B-UnspecificTherapy", "I-PlanHeader", "B-CancerDx", "B-UnspecificTherapy", "I-PlanHeader", "B-Location", "B-Age", "B-CancerDx", "B-CancerDx", "B-TestResult", "B-Age", "B-UnspecificTherapy", "I-HormonalTherapy", "I-HormonalTherapy", "I-HormonalTherapy", "O", "I-PlanHeader", "O", "B-Location", "B-Location", "B-TestResult", "B-UnspecificTherapy", "B-CancerDx", "B-CancerDx", "I-HormonalTherapy", "B-UnspecificTherapy", "B-Age", "O", "B-TestResult", "I-PlanHeader", "B-TestResult", "I-PlanHeader", "I-PlanHeader", "I-PlanHeader", "B-TestResult", "B-TestResult", "I-HormonalTherapy", "I-HormonalTherapy", "B-UnspecificTherapy", "I-HormonalTherapy", "I-PlanHeader", "I-HormonalTherapy", "B-Location", "I-HormonalTherapy", "B-Age", "B-CancerDx", "O", "B-UnspecificTherapy", "B-TestResult", "O", "O", "B-TestResult", "I-HormonalTherapy", "O", "B-TestResult", "B-Age", "B-CancerDx", "B-TestResult", "B-Age", "B-UnspecificTherapy", "I-HormonalTherapy", "I-HormonalTherapy", "O", "O", "B-CancerDx", "B-Age", "B-Location", "I-PlanHeader", "I-HormonalTherapy", "B-TestResult", "B-Location", "B-Age", "I-HormonalTherapy", "B-Age")

  "NerEvaluator.evaluateNer" should "create a valid Map" in {
    val getOutput = NerEvaluator.evaluateNer(ResourceHelper.spark, ground_truth, predictions)
    val expectedMap = Map("accuracy" -> (None,None,19.101124.toFloat), "micro" -> (18.88889.toFloat,19.101124.toFloat,18.994415.toFloat), "macro" -> (18.344017.toFloat,17.61905.toFloat,17.61905.toFloat), "weighted" -> (20.710169.toFloat, 19.101124.toFloat, 19.328587.toFloat))
    assert (expectedMap.equals(getOutput._1) )
  }

  "NerEvaluator.evaluateNer" should "also work for dataframe and token_level" in {
    val getOutput = NerEvaluator.evaluateNer(ResourceHelper.spark, ground_truth, predictions, mode="token_level", outputDf = true)
    val expectedRow = Row("B-Location", 30.769232.toFloat, 26.666668.toFloat, 28.57143.toFloat, 15)
    val outputRow = getOutput._3.getOrElse(0).asInstanceOf[DataFrame].first()
    assert(outputRow.equals(expectedRow))
  }

  "NerEvaluator.evaluateNer" should "work for all parameters" in {
    val defaultExample = NerEvaluator.evaluateNer(ResourceHelper.spark, ground_truth, predictions)
    print("default example (no output df, mode is entity level so we are removing IOB tagging, and we are using proportions instead of percentages):\n")
    print("overall (precision, recall, f1) for each method of averaging: " + defaultExample._1 + "\n")
    print("(precision, recall, f1, support) for each label: " + defaultExample._2+ "\n")
    print("dataframe (None because we didn't do the outputDf flag): " + defaultExample._3+ "\n")

    print("\n")
    val percentTokenAndDfExample = NerEvaluator.evaluateNer(ResourceHelper.spark, ground_truth, predictions, percent = true, outputDf = true, mode = "token_level")
    print("percent=true, mode='token_level', and outputDf=true example:\n")
    print("overall (precision, recall, f1) for each method of averaging: " + percentTokenAndDfExample._1+ "\n")
    print("(precision, recall, f1, support) for each label: " + percentTokenAndDfExample._2+ "\n")
    print("dataframe for labels: \n")
    percentTokenAndDfExample._3.get.show(10)
    print("Recall for Location: "  + percentTokenAndDfExample._2.getOrElse("B-Location", (0,0,0,0)).asInstanceOf[(Float, Float, Float, Int)]._2.toString + "\n")
    print("Overall Micro f1: "  + percentTokenAndDfExample._1.getOrElse( "micro", (0,0,0)).asInstanceOf[(Float, Float, Float)]._3.toString + "\n")
  }


}
