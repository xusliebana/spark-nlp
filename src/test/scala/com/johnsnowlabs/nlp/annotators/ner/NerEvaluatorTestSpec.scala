package com.johnsnowlabs.nlp.annotators.ner

import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.apache.spark.sql.Row
import org.scalatest.FlatSpec

class NerEvaluatorTestSpec extends FlatSpec{


  val ground_truth = Seq("B-Location", "B-Age", "B-TestResult", "B-UnspecificTherapy", "B-Age", "O", "B-CancerDx", "O", "O", "O", "B-UnspecificTherapy", "B-UnspecificTherapy", "I-HormonalTherapy", "B-Age", "I-PlanHeader", "I-HormonalTherapy", "B-Location", "B-Location", "I-HormonalTherapy", "B-Age", "B-UnspecificTherapy", "O", "B-Age", "B-CancerDx", "O", "I-PlanHeader", "B-Location", "B-Age", "I-PlanHeader", "B-TestResult", "B-Age", "I-HormonalTherapy", "B-Age", "B-TestResult", "B-UnspecificTherapy", "B-Location", "O", "B-CancerDx", "B-TestResult", "B-CancerDx", "I-PlanHeader", "B-Age", "B-Age", "O", "I-HormonalTherapy", "B-UnspecificTherapy", "B-CancerDx", "B-TestResult", "O", "B-TestResult", "B-CancerDx", "B-Location", "B-Age", "O", "B-TestResult", "B-Age", "I-HormonalTherapy", "B-Location", "B-UnspecificTherapy", "B-CancerDx", "B-UnspecificTherapy", "B-TestResult", "I-HormonalTherapy", "I-PlanHeader", "I-PlanHeader", "B-CancerDx", "I-HormonalTherapy", "I-HormonalTherapy", "B-Location", "I-HormonalTherapy", "B-TestResult", "I-PlanHeader", "B-CancerDx", "I-PlanHeader", "B-Age", "I-PlanHeader", "B-Location", "B-Location", "B-CancerDx", "B-Location", "B-Location", "B-Age", "B-CancerDx", "B-CancerDx", "B-CancerDx", "B-TestResult", "I-HormonalTherapy", "B-TestResult", "B-Location", "I-HormonalTherapy", "B-Age", "O", "B-Location", "B-TestResult", "B-CancerDx", "B-CancerDx", "B-Location", "I-PlanHeader", "I-PlanHeader", "B-Age")
  val predictions = Seq("I-HormonalTherapy", "B-TestResult", "B-TestResult", "B-Location", "B-Location", "B-CancerDx", "B-CancerDx", "I-HormonalTherapy", "I-HormonalTherapy", "B-Age", "B-Location", "O", "B-TestResult", "I-PlanHeader", "B-Location", "B-TestResult", "B-Location", "B-TestResult", "B-CancerDx", "B-Location", "B-Location", "B-CancerDx", "I-PlanHeader", "B-UnspecificTherapy", "B-UnspecificTherapy", "I-HormonalTherapy", "B-UnspecificTherapy", "I-PlanHeader", "B-CancerDx", "B-UnspecificTherapy", "I-PlanHeader", "B-Location", "B-Age", "B-CancerDx", "B-CancerDx", "B-TestResult", "B-Age", "B-UnspecificTherapy", "I-HormonalTherapy", "I-HormonalTherapy", "I-HormonalTherapy", "O", "I-PlanHeader", "O", "B-Location", "B-Location", "B-TestResult", "B-UnspecificTherapy", "B-CancerDx", "B-CancerDx", "I-HormonalTherapy", "B-UnspecificTherapy", "B-Age", "O", "B-TestResult", "I-PlanHeader", "B-TestResult", "I-PlanHeader", "I-PlanHeader", "I-PlanHeader", "B-TestResult", "B-TestResult", "I-HormonalTherapy", "I-HormonalTherapy", "B-UnspecificTherapy", "I-HormonalTherapy", "I-PlanHeader", "I-HormonalTherapy", "B-Location", "I-HormonalTherapy", "B-Age", "B-CancerDx", "O", "B-UnspecificTherapy", "B-TestResult", "O", "O", "B-TestResult", "I-HormonalTherapy", "O", "B-TestResult", "B-Age", "B-CancerDx", "B-TestResult", "B-Age", "B-UnspecificTherapy", "I-HormonalTherapy", "I-HormonalTherapy", "O", "O", "B-CancerDx", "B-Age", "B-Location", "I-PlanHeader", "I-HormonalTherapy", "B-TestResult", "B-Location", "B-Age", "I-HormonalTherapy", "B-Age")

  "NerEvaluator.evaluateNer" should "create a valid Map" in {
    val getOutput = NerEvaluator.evaluateNER(ResourceHelper.spark, ground_truth, predictions)
    val expectedMap = Map("total precision"->18.88889, "total recall" ->19.101124, "total f1" ->18.994415)
    for ((k,v) <- getOutput._1) assert(expectedMap(k).toFloat == v.toFloat)
  }

  "NerEvaluator.evaluateNer" should "also work for token_level" in {
    val getOutput = NerEvaluator.evaluateNER(ResourceHelper.spark, ground_truth, predictions, mode="token_level")
    val expectedRow = Row("B-Location", 30.769232.toFloat, 26.666668.toFloat, 28.57143.toFloat, 15)
    val outRow = getOutput._2.first()
    assert(outRow.equals(expectedRow))
  }

}
