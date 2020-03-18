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
      "i like burgers",
      "if it looks like a duck, swims like a duck, and quacks like a duck, then it probably is a duck",
      "we can only see a short distance ahead but we can see plenty there that needs to be done"
    ).toDF("text")

    val albert_model_path = "/home/loan/Documents/JohnSnowLabs/spark-nlp-training/python/tensorflow/albert/exported_albert/albert/content/AlbertModel"
    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector().setInputCols("document")
      .setOutputCol("sentence")


    val albertSavedModel = AlbertEmbeddings.loadSavedModel(albert_model_path, SparkNLP.start())
      .setBatchSize(2)
      .setPoolingLayer("token_embeddings")
      .setInputCols(Array("document"))
      .setOutputCol("embeddings")


    albertSavedModel.write.overwrite().save("./tmp_albert_tf")

    val embeddings = AlbertEmbeddings.load("./tmp_albert_tf")

    val pipeline = new Pipeline().setStages(Array(
      document,
      sentence,
      embeddings
    ))

    val albertDDD = pipeline.fit(data).transform(data)
    print("debug")
    albertDDD.show(10, false)
  }


}
