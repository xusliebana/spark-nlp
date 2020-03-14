package com.johnsnowlabs.ml.tensorflow
import com.johnsnowlabs.nlp.annotators.common._
import scala.collection.JavaConverters._

/**
  * This class is used to calculate ELMO embeddings for For Sequence Batches of TokenizedSentences.
  *
  * @param tensorflow       SentencePiece Model wrapper with TensorFlow Wrapper
  * @param batchSize        size of batch
  * @param configProtoBytes Configuration for TensorFlow session
  */

class TensorflowSentencePiece(val tensorflow: TensorflowWrapper,
                              batchSize: Int,
                              configProtoBytes: Option[Array[Byte]] = None,
                              maxTokenLength: Int = 25, // todo fix magic numbr
                              idToTokenMap: scala.collection.mutable.Map[Int, String]
                             ) extends Serializable {
  //Todo make keys "module/node" pattern
  private val inputStringsKey = "sentenceInput"
  private val outputIdsKey = "ids"
  private val outputSeqLenKey = "module/input_ids"
  private val inputIdsKey = "tokensInput"
  private val outputTokensKey = "tokensOut"

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
      .feed(inputStringsKey, sentenceTensors)
      .fetch(outputIdsKey)
    //      .fetch(outputSeqLenKey)  we can remove ths  by just getting the length in java..


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
  def getTokensForIdsMapBased(batch: Seq[Seq[Int]]): Seq[WordpieceTokenizedSentence] = {

    batch.map { sentence =>
      sentence.map {
        tokenId =>
          TokenPiece(wordpiece = idToTokenMap(tokenId),
            token = idToTokenMap(tokenId),
            pieceId = tokenId,
            isWordStart = true,
            begin = -1,
            end = -1
          )
      }
    }.map(sentenceTokenPieces => WordpieceTokenizedSentence(sentenceTokenPieces.toArray))
    //
    //
    //    print("debug")
    //
    //    batch.flatten.zip(decoded).map { case (tokenId, token) =>
    //      Array(TokenPiece(wordpiece = decoded.toString(),
    //        token = token,
    //        pieceId =tokenId ,
    //        isWordStart = true,
    //        begin = -1,
    //        end = -1
    //      ))
    //
    //    }.map(tokens => WordpieceTokenizedSentence(tokens))
    //
    //    Seq[WordpieceTokenizedSentence]()
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
      .feed(inputIdsKey, sentenceTensors)
      .fetch(outputTokensKey)
    //      .fetch("seq_len")  we can remove ths  by just getting the length in java..


    val outs = runner.run().asScala
    // only real efficient way to allocate bytes is by looking them up in the dctionary. The we know the actualy tokensize.
    // But then we could also just do all decoding dictionary based..

    val tokensBytes = TensorResources.extractBytes(outs(0), maxTokenLength = maxTokenLength) // Depends on the fetch order!
    //    val seq_lens = TensorResources.extractInts(outs(1))
    val string = new String(tokensBytes, "UTF-8");

    print("debug")

    tensors.clearSession(outs)
    tensors.clearTensors()
    WordpieceTokenizedSentence

    val res = batch.zip(Seq(tokensBytes)).map { case (tokenId, tokenBytes) =>
      Array(TokenPiece(wordpiece = new String(tokensBytes, "UTF-8"),
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