package com.johnsnowlabs.nlp.annotators.html

import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp._
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest.Matchers._
import org.scalatest._

import scala.language.reflectiveCalls

class HtmlParserTestSpec extends FlatSpec {



  def rawDataset(content: String*): Dataset[Row] = {
    import SparkAccessor.spark.implicits._
    SparkAccessor.spark.sparkContext.parallelize(content).toDS().toDF("text")
  }

    val text =
      """ <html>
        |  <head>
        |   <title>
        |    The Dormouse's story
        |   </title>
        |  </head>
        |  <body>
        |   <p class="title">
        |    <b>
        |     The Dormouse's story title
        |    </b>
        |   </p>
        |   <p class="story">
        |    Once upon a time there were three little sisters; and their names were
        |    <a class="sister" href="http://example.com/elsie" id="link1">
        |     Elsie
        |    </a>
        |    ,
        |    <a class="sister" href="http://example.com/lacie" id="link2">
        |     Lacie
        |    </a>
        |    and
        |    <a class="sister" href="http://example.com/tillie" id="link3">
        |     Tillie
        |    </a>
        |    ; and they lived at the bottom of a well.
        |   </p>
        |   <p class="story">
        |     OtherStory
        |   </p>
        |  </body>
        | </html>""".stripMargin


  "A HtmlTransformer" should "return the content of title" in {
    val df = DataBuilder.basicDataBuild(text)
    val htmlParser = new HtmlParser()
      .setInputCols(Array("document"))
      .setOutputCol("htmlTag")
      .setTag("html.head.title")
    val df2 = htmlParser.transform(df)
    val htmltTags = df2
      .select("htmlTag")
      .collect()
      .flatMap { _.getSeq[Row](0) }
      .map { Annotation(_) }
    "\n    The Dormouse's story\n   " should equal (htmltTags.head.result)
    27 should equal (htmltTags.head.begin)
    55 should equal (htmltTags.head.`end`)
    1 should equal (htmltTags.size)

  }

  "A HtmlTransformer" should "return the whole tag of title" in {
    val df = DataBuilder.basicDataBuild(text)
    val htmlParser = new HtmlParser()
      .setInputCols(Array("document"))
      .setOutputCol("htmlTag")
      .setTag("html.head.title")
      .setSelector("all")
    val df2 = htmlParser.transform(df)
    val htmltTags = df2
      .select("htmlTag")
      .collect()
      .flatMap { _.getSeq[Row](0) }
      .map { Annotation(_) }
    "<title>\n    The Dormouse's story\n   </title>" should equal (htmltTags.head.result)
    20 should equal (htmltTags.head.begin)
    63 should equal (htmltTags.head.`end`)
    1 should equal (htmltTags.size)

  }


  "A HtmlTransformer" should "return not annotator" in {
    val df = DataBuilder.basicDataBuild(text)
    val htmlParser = new HtmlParser()
      .setInputCols(Array("document"))
      .setOutputCol("html")
      .setTag("notFound")
    val df2 = htmlParser.transform(df)
    val htmltTags = df2
      .select("html")
      .collect()
      .flatMap { _.getSeq[Row](0) }
      .map { Annotation(_) }
    0 should equal (htmltTags.size)
  }

  "A HtmlTransformer" should "return 3 p classes" in {
    val df = DataBuilder.basicDataBuild(text)
    val htmlParser = new HtmlParser()
      .setInputCols(Array("document"))
      .setOutputCol("htmlTag")
      .setTag("notFound")
    val df2 = htmlParser.transform(df)
    val htmltTags = df2
      .select("html")
      .collect()
      .flatMap { _.getSeq[Row](0) }
      .map { Annotation(_) }
    0 should equal (htmltTags.size)
  }


  "A HtmlTransformer" should "works in a pipeline" in {

    val data = rawDataset(text)
    val documentAssembler: DocumentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")


    val htmlParser = new HtmlParser()
      .setInputCols(Array("document"))
      .setOutputCol("html")
      .setTag("html.head.title")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("html"))
      .setOutputCol("token")

    val finisher = new Finisher(+)
      .setInputCols("token")
      .setOutputAsArray(false)
      .setAnnotationSplitSymbol("@")
      .setValueSplitSymbol("#")

    val pipeline: Pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        new Pipeline().setStages(Array(documentAssembler)),
        htmlParser,
        tokenizer,
        finisher
      ))

    val resultDataset = pipeline.fit(data).transform(data)
    "The@Dormouse's@story" should equal(resultDataset.select("finished_token").collect()(0)(0))




  }

}