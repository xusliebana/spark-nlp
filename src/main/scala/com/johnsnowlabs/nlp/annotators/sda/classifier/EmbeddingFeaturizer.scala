package com.johnsnowlabs.nlp.annotators.sda.classifier

import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Row}

class EmbeddingFeaturizer(override val uid: String)
  extends Transformer with HasInputCol with HasOutputCol with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("EMBEDDING_FEATURIZER"))

  def setInputCol(value: String): this.type = set(inputCol, value)

  def setOutputCol(value: String): this.type = set(outputCol, value)

  setDefault(
    inputCol -> "embeddings",
    outputCol -> "centroid"
  )

  override def transform(dataset: Dataset[_]): DataFrame = {
    val embeddingCol = dataset($(inputCol))
    val featureVectors = EmbeddingFeaturizer.averageEmbeddingsUDF(embeddingCol)
    dataset.withColumn($(outputCol), featureVectors)
  }

  override def transformSchema(schema: StructType): StructType = {
    if (schema.fieldNames.contains($(outputCol))) {
      throw new IllegalArgumentException(s"Output column ${$(outputCol)} already exists.")
    }
    val outputField = new AttributeGroup($(outputCol))
    val outputFields = schema.fields :+ outputField.toStructField()
    StructType(outputFields)
  }

  override def copy(extra: ParamMap): EmbeddingFeaturizer = defaultCopy(extra)
}

object EmbeddingFeaturizer extends DefaultParamsReadable[EmbeddingFeaturizer] {
  def averageEmbeddings(annotations: Seq[Annotation]): Vector = {
    val embeddings: Seq[Array[Float]] = annotations
      .filter(a => a.annotatorType == AnnotatorType.WORD_EMBEDDINGS).map(_.embeddings)
    val dim = embeddings.head.length
    val centroid = Array.fill(dim)(0.0)
    val iter = 0 until dim
    embeddings.foreach {
      arr =>
        iter.foreach {
          i =>
            centroid(i) += arr(i)
        }
    }
    iter.foreach {
      i =>
        centroid(i) /= embeddings.size
    }
    Vectors.dense(centroid)
  }

  val averageEmbeddingsUDF: UserDefinedFunction =
    udf {
      annotationContent: Seq[Row] =>
        val annotations = annotationContent.map {
          anno =>
            Annotation(anno)
        }
        averageEmbeddings(annotations)
    }

  override def load(path: String): EmbeddingFeaturizer = super.load(path)
}