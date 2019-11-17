package com.johnsnowlabs.nlp.annotators.sda.classifier

import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.shared.HasOutputCol
import org.apache.spark.ml.param.{DoubleParam, Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.types.{StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row, functions}

class PredictionConverter(override val uid: String)
  extends Transformer with HasOutputCol with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("PREDICTION_CONVERTER"))

  def setOutputCol(value: String): this.type = set(outputCol, value)

  val predictionCol: Param[String] = new Param[String](this, "predictionCol", "the column containing the predicted class probabilities")

  def getPredictionCol: String = $(predictionCol)

  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  val documentCol: Param[String] = new Param[String](this, "documentCol", "the column containing the document annotations")

  def getDocumentCol: String = $(documentCol)

  def setDocumentCol(value: String): this.type = set(documentCol, value)

  val positiveClass: Param[String] = new Param[String](this, "positiveClass", "the label for the positive class")

  def getPositiveClass: String = $(positiveClass)

  def setPositiveClass(value: String): this.type = set(positiveClass, value)

  val negativeClass: Param[String] = new Param[String](this, "negativeClass", "the label for the negative class")

  def getNegativeClass: String = $(negativeClass)

  def setNegativeClass(value: String): this.type = set(negativeClass, value)

  val threshold: DoubleParam = new DoubleParam(this, "threshold", "the threshold above which is positive, below negative")

  def getThreshold: Double = $(threshold)

  def setThreshold(value: Double): this.type = set(threshold, value)

  setDefault(
    predictionCol -> "prediction",
    documentCol -> "document",
    positiveClass -> "positive",
    negativeClass -> "negative",
    threshold -> 0.5
  )

  override def transform(dataset: Dataset[_]): DataFrame = {
    val buildAnnotationUDF: UserDefinedFunction = functions.udf(buildAnnotation _)
    val sentimentCol = buildAnnotationUDF(dataset($(predictionCol)), dataset($(documentCol)))
    dataset.withColumn($(outputCol), sentimentCol)
  }

  override def transformSchema(schema: StructType): StructType = {
    if (schema.fieldNames.contains($(outputCol))) {
      throw new IllegalArgumentException(s"Output column ${$(outputCol)} already exists.")
    }
    val outputField = StructField($(outputCol), new StructType())
    val outputFields = schema.fields :+ outputField
    StructType(outputFields)
  }


  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  def buildAnnotation(score: Double, annotations: Seq[Row]): Annotation = {
    val document = annotations.view.map(Annotation(_)).find(_.annotatorType == AnnotatorType.DOCUMENT) match {
      case Some(a) => a
      case _ => throw new IllegalArgumentException(s"no ${AnnotatorType.DOCUMENT} found")
    }
    val label = if (score >= 0.5) {
      $(positiveClass)
    } else if (score < 0.5) {
      $(negativeClass)
    } else {
      "na"
    }
    Annotation(AnnotatorType.SENTIMENT, document.begin, document.end, label, Map("confidence" -> f"$score%.6f"))
  }
}

object PredictionConverter extends DefaultParamsReadable[PredictionConverter] {
  override def load(path: String): PredictionConverter = super.load(path)
}