package com.johnsnowlabs.ml.tensorflow

import com.johnsnowlabs.nlp.annotators.common.WordpieceTokenized.annotatorType
import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}
import com.johnsnowlabs.nlp.annotators.common._

import scala.collection.JavaConverters._
import scala.collection.Map

/**
  * This class is used to calculate ELMO embeddings for For Sequence Batches of TokenizedSentences.
  *
  * https://tfhub.dev/google/elmo/3
  * * word_emb: the character-based word representations with shape [batch_size, max_length, 512].  == word_emb
  * * lstm_outputs1: the first LSTM hidden state with shape [batch_size, max_length, 1024]. === lstm_outputs1
  * * lstm_outputs2: the second LSTM hidden state with shape [batch_size, max_length, 1024]. === lstm_outputs2
  * * elmo: the weighted sum of the 3 layers, where the weights are trainable. This tensor has shape [batch_size, max_length, 1024]  == elmo
  *
  * @param tensorflow       Elmo Model wrapper with TensorFlow Wrapper
  * @param batchSize        size of batch
  * @param configProtoBytes Configuration for TensorFlow session
  */

class TensorflowSentencePiece(val tensorflow: TensorflowWrapper,
                              batchSize: Int,
                              configProtoBytes: Option[Array[Byte]] = None,
                              EOSToken: String = "[TODO]",
                              maxTokenLength: Int = 100
                             ) extends Serializable {

  /**
    * Calculate the embeddings for a sequence of Tokens and create WordPieceEmbeddingsSentence objects from them
    * It corrosponds to the encode() method of the tf_sentencepiece module
    *
    * @param sentences A sequence of Tokenized Sentences for which embeddings will be calculated
    * @return A Seq of WordpieceEmbeddingsSentence, one element for each input sentence
    */
  def calculateTokensAndIds(sentences: Seq[String]): Seq[WordpieceTokenizedSentence] = {

    /*Run embeddings calculation by batches*/
    //    sentences.zipWithIndex.grouped(batchSize).flatMap { batch =>
    val vectors = tokenizeAndGetIds(sentences)

    //        }
    /*Combine tokens and sentences  and their calculated tokens and IDs TODO*/

    vectors
  }

  /**
    * This function will tokenize each sentence and find the ID's for each of the tokens using a Sentence Piece model.
    * It corrosponds to the piece_to_id() method of the tf_sentencepiece module
    *
    * @param batch The sentences for which we calculate tokens and ids
    * @return The Embeddings Vector. For each Seq Element we have a Sentence, and for each sentence we have an Array for each of its words. Each of its words gets a float array to represent its Embeddings
    */
  def tokenizeAndGetIds(batch: Seq[String]): Seq[WordpieceTokenizedSentence] = {

    val tensors = new TensorResources()

    /* Actual size of each sentence to skip padding in the TF model */

    val sentencesBytes = batch.map { sentence =>
      sentence.getBytes("UTF-8")
    }.toArray

    val sentenceTensors = tensors.createTensor(sentencesBytes)
    val runner = tensorflow.getSession(configProtoBytes = configProtoBytes).runner

    runner
      .feed("sentenceInput", sentenceTensors)
      .fetch("ids")
    //      .fetch("seq_len")  we can remove ths  by just getting the length in java..


    val outs = runner.run().asScala
    val ids = TensorResources.extractInts(outs(0)) // Depends on the fetch order!
    //    val seq_lens = TensorResources.extractInts(outs(1))

    print("debug")

    tensors.clearSession(outs)
    tensors.clearTensors()
    WordpieceTokenizedSentence

    batch.zip(ids).map { case (sentence, id) =>
      Array(TokenPiece(wordpiece = ids.mkString(","), //inefficient hack
        token = "#",
        pieceId = -1,
        isWordStart = true,
        begin = -1,
        end = -1
      ))

    }.map(tokens => WordpieceTokenizedSentence(tokens))

  }


  /**
    * This function will tokenize each sentence and find the ID's for each of the tokens using a Sentence Piece model.
    *
    * @param batch The sentences Ids for which we calculate original string tokens
    * @return The Embeddings Vector. For each Seq Element we have a Sentence, and for each sentence we have an Array for each of its words. Each of its words gets a float array to represent its Embeddings
    */
  def getTokensForIds(batch: Seq[Seq[Int]]): Seq[WordpieceTokenizedSentence] = {
    // this could be done just using a dict..
    val tensors = new TensorResources()

    /* Actual size of each sentence to skip padding in the TF model */

    val sentencesBytes = batch.map { sentence =>
      sentence.map(id => id).toArray
    }.toArray

    val sentenceTensors = tensors.createTensor(sentencesBytes)
    val runner = tensorflow.getSession(configProtoBytes = configProtoBytes).runner

    runner
      .feed("tokensInput", sentenceTensors)
      .fetch("tokensOut")
    //      .fetch("seq_len")  we can remove ths  by just getting the length in java..


    val outs = runner.run().asScala
    val amountOftokens = 1000
    val tokensBytes = TensorResources.extractBytes(outs(0), maxTokenLength = maxTokenLength, amountOfTokens = amountOftokens) // Depends on the fetch order!
    //    val seq_lens = TensorResources.extractInts(outs(1))
    val string = new String(tokensBytes, "UTF-8");

    print("debug")

    tensors.clearSession(outs)
    tensors.clearTensors()
    WordpieceTokenizedSentence

    val res = batch.zip(Seq(tokensBytes)).map { case (tokenId, tokenBytes) =>
      Array(TokenPiece(wordpiece = new String(tokensBytes, "UTF-8"), // "lol".mkString(","), //inefficient hack
        token = batch(0).mkString(","),
        pieceId = -1,
        isWordStart = true,
        begin = -1,
        end = -1
      ))

    }.map(tokens => WordpieceTokenizedSentence(tokens))
    print(res)
    res
  }


}