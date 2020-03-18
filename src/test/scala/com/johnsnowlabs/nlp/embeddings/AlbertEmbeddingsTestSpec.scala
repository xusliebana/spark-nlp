package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.nlp.SparkNLP
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.util.io.ResourceHelper.spark.implicits._
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions.{explode, size}
import org.scalatest._

class AlbertEmbeddingsTestSpec extends FlatSpec {
  "Albert Embeddings" should "generate annotations" in {
    System.out.println("Working Directory = " + System.getProperty("user.dir"))
    val data = Seq(
      "i like burger",
      "if it looks like a duck, swims like a duck, and quacks like a duck, then it probably is a duck",
      "we can only see a short distance ahead but we can see plenty there that needs to be done"
    ).toDF("text")

    val albert_model_path = "/home/loan/Documents/JohnSnowLabs/spark-nlp-training/python/tensorflow/albert/exported_albert/albert/content/AlbertModel"
    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector().setInputCols("document")
      .setOutputCol("sentence")


    val xlnetSavedModel = AlbertEmbeddings.loadSavedModel(albert_model_path, SparkNLP.start())
      .setBatchSize(2)
      .setPoolingLayer("token_embeddings")
      .setInputCols(Array("document"))
      .setOutputCol("embeddings")


    xlnetSavedModel.write.overwrite().save("./tmp_albert_tf")

    val embeddings = AlbertEmbeddings.load("./tmp_albert_tf")

    val pipeline = new Pipeline().setStages(Array(
      document,
      sentence,
      embeddings
    ))

    val xlnetDDD = pipeline.fit(data).transform(data)
    xlnetDDD.show(10, false)

    //, [0.73741215, 0.12575178, -0.969205, -0.29848674, -0.5927709, -1.6572264, 0.3679387, 0.27600533, -0.19157335, 0.
    // [-0.78455746, -1.2591217, 0.15253855, -0.15794446, 0.92008966, 1.2331969, 1.5931642, -0.20098667, -0.41274026, 0.5332916, -0
    // [1.1833516, -0.4436788, 1.8499879, -1.6329402, -1.1829668, -1.563963, -0.31045082, -0.97847515, -0.8943779,


    //    , [0.07470194, 0.9456775, 0.57883626, 0.3362052, 0.23846953, 1.5805937, -0.1546841, -0.4855847,
  }


}
