import sbtassembly.MergeStrategy

/** Version variables */
val sparkVer = "2.4.3"
val scalaVer = "2.11.12"
val scalaTestVersion = "3.0.0"

val awsSDKVer = "1.11.603"
val configVer = "1.3.0"
val greexVer = "1.0"
val hadoopVer = "3.2.0"
val jaiCoreVer = "1.1.3"
val jbig2ImageioVer = "3.0.2"
val json4sVer = "3.5.3"
val junitVer = "4.10"
val liblevenshteinVer = "3.0.0"
val pdfboxVer = "2.0.13"
val rocksdbVer = "5.17.2"
val tensorflowVer = "1.12.0"
val tess4jVer = "4.2.1"
val trove4jVer = "3.0.3"

/** Spark-Package attributes */
organization:= "com.johnsnowlabs.nlp"
spName in ThisBuild := "JohnSnowLabs/spark-nlp"
version := "2.2.1"
scalaVersion in ThisBuild := scalaVer
sparkVersion in ThisBuild := sparkVer

licenses  += "Apache-2.0" -> url("http://opensource.org/licenses/Apache-2.0")

val is_gpu = System.getProperty("is_gpu","false")

/** Package attributes */
if(is_gpu.equals("false")){
  name := "spark-nlp"
}else{
  name:="spark-nlp-gpu"
}

sparkComponents in ThisBuild ++= Seq("mllib")
spIncludeMaven in ThisBuild:= false
spAppendScalaVersion := false

resolvers in ThisBuild += "Maven Central" at "http://central.maven.org/maven2/"
resolvers in ThisBuild += "Spring Plugins" at "http://repo.spring.io/plugins-release/"

assemblyOption in assembly := (assemblyOption in assembly).value.copy(
  includeScala = false
)

credentials += Credentials(Path.userHome / ".ivy2" / ".sbtcredentials")

ivyScala := ivyScala.value map {
  _.copy(overrideScalaVersion = true)
}

/** Bintray settings */
bintrayRepository := "spark-nlp"
bintrayOrganization:= Some("johnsnowlabs")
sonatypeProfileName := "com.johnsnowlabs"
bintrayPackageLabels := Seq("nlp", "nlu",
  "natural-language-processing", "natural-language-understanding",
  "spark", "spark-ml", "pyspark", "machine-learning",
  "named-entity-recognition", "sentiment-analysis", "lemmatizer", "spell-checker",
  "tokenizer", "stemmer", "part-of-speech-tagger", "annotation-framework")

publishTo := Some(
  if (isSnapshot.value)
    Opts.resolver.sonatypeSnapshots
  else
    Opts.resolver.sonatypeStaging
)

homepage:= Some(url("https://nlp.johnsnowlabs.com"))

scmInfo:= Some(
  ScmInfo(
    url("https://github.com/JohnSnowLabs/spark-nlp"),
    "scm:git@github.com:JohnSnowLabs/spark-nlp.git"
  )
)

developers in ThisBuild:= List(
  Developer(id="saifjsl", name="Saif Addin", email="saif@johnsnowlabs.com", url=url("https://github.com/saifjsl")),
  Developer(id="maziyarpanahi", name="Maziyar Panahi", email="maziyar@johnsnowlabs.com", url=url("https://github.com/maziyarpanahi")),
  Developer(id="albertoandreottiATgmail", name="Alberto Andreotti", email="alberto@pacific.ai", url=url("https://github.com/albertoandreottiATgmail")),
  Developer(id="danilojsl", name="Danilo Burbano", email="danilo@johnsnowlabs.com", url=url("https://github.com/danilojsl")),
  Developer(id="rohit13k", name="Rohit Kumar", email="rohit@johnsnowlabs.com", url=url("https://github.com/rohit13k")),
  Developer(id="aleksei-ai", name="Aleksei Alekseev", email="aleksei@pacific.ai", url=url("https://github.com/aleksei-ai")),
  Developer(id="showy", name="Eduardo Muñoz", email="eduardo@johnsnowlabs.com", url=url("https://github.com/showy"))
)

target in Compile in doc := baseDirectory.value / "docs/api"

lazy val ocrDependencies = Seq(
  "net.sourceforge.tess4j" % "tess4j" % tess4jVer
    exclude("org.slf4j", "slf4j-log4j12")
    exclude("org.apache.logging", "log4j"),
  "org.apache.pdfbox" % "pdfbox" % pdfboxVer,
  "org.apache.pdfbox" % "jbig2-imageio" % jbig2ImageioVer,
  "javax.media.jai" % "com.springsource.javax.media.jai.core" % jaiCoreVer
)

lazy val analyticsDependencies = Seq(
  "org.apache.spark" %% "spark-core" % sparkVer % "provided",
  "org.apache.spark" %% "spark-mllib" % sparkVer % "provided"
)

lazy val utilDependencies = Seq(
  "com.typesafe" % "config" % configVer,
  "org.rocksdb" % "rocksdbjni" % rocksdbVer,
  "org.apache.hadoop" % "hadoop-aws" %  hadoopVer
    exclude("com.fasterxml.jackson.core", "jackson-annotations")
    exclude("com.fasterxml.jackson.core", "jackson-databind")
    exclude("com.fasterxml.jackson.core", "jackson-core")
    exclude("commons-configuration","commons-configuration")
    exclude("com.amazonaws","aws-java-sdk-bundle")
    exclude("org.apache.hadoop" ,"hadoop-common"),
  "com.amazonaws" % "aws-java-sdk-core" % awsSDKVer
    exclude("com.fasterxml.jackson.core", "jackson-annotations")
    exclude("com.fasterxml.jackson.core", "jackson-databind")
    exclude("com.fasterxml.jackson.core", "jackson-core")
    exclude("commons-configuration","commons-configuration"),
  "com.amazonaws" % "aws-java-sdk-s3" % awsSDKVer,
  "org.rocksdb" % "rocksdbjni" % rocksdbVer,
  "com.github.universal-automata" % "liblevenshtein" % liblevenshteinVer
    exclude("com.google.guava", "guava")
    exclude("org.apache.commons", "commons-lang3"),
  "com.navigamez" % "greex" % greexVer,
  "org.json4s" %% "json4s-ext" % json4sVer
)

lazy val typedDependencyParserDependencies = Seq(
  "net.sf.trove4j" % "trove4j" % trove4jVer,
  "junit" % "junit" % junitVer % Test // Why is this here and not in the testDependencies?
)
val tensorflowDependencies: Seq[sbt.ModuleID] =
  if (is_gpu.equals("true"))
    Seq(
      "org.tensorflow" % "libtensorflow" % tensorflowVer,
      "org.tensorflow" % "libtensorflow_jni_gpu" % tensorflowVer
    )
  else
    Seq(
      "org.tensorflow" % "tensorflow" % tensorflowVer
    )

lazy val testDependencies = Seq(
  "org.scalatest" %% "scalatest" % scalaTestVersion % "test"
)

lazy val root = (project in file("."))
  .settings(
    libraryDependencies ++=
      analyticsDependencies ++
        testDependencies ++
        utilDependencies ++
        tensorflowDependencies++
        typedDependencyParserDependencies
  )


val ocrMergeRules: String => MergeStrategy  = {
  case "versionchanges.txt" => MergeStrategy.discard
  case "StaticLoggerBinder" => MergeStrategy.discard
  case PathList("META-INF", fileName)
    if List("NOTICE", "MANIFEST.MF", "DEPENDENCIES", "INDEX.LIST").contains(fileName) || fileName.endsWith(".txt")
  => MergeStrategy.discard
  case PathList("META-INF", "services", _ @ _*)  => MergeStrategy.first
  case PathList("META-INF", xs @ _*)  => MergeStrategy.first
  case PathList("org", "apache", _ @ _*)  => MergeStrategy.first
  case PathList("apache", "commons", "logging", "impl",  xs @ _*)  => MergeStrategy.discard
  case _ => MergeStrategy.deduplicate
}

val evalMergeRules: String => MergeStrategy  = {
  case "versionchanges.txt" => MergeStrategy.discard
  case "StaticLoggerBinder" => MergeStrategy.discard
  case PathList("META-INF", fileName)
    if List("NOTICE", "MANIFEST.MF", "DEPENDENCIES", "INDEX.LIST").contains(fileName) || fileName.endsWith(".txt")
  => MergeStrategy.discard
  case PathList("META-INF", "services", _ @ _*)  => MergeStrategy.first
  case PathList("META-INF", xs @ _*)  => MergeStrategy.first
  case PathList("org", "apache", "spark", _ @ _*)  => MergeStrategy.discard
  case PathList("apache", "commons", "logging", "impl",  xs @ _*)  => MergeStrategy.discard
  case _ => MergeStrategy.deduplicate
}

assemblyMergeStrategy in assembly := {
  case PathList("apache.commons.lang3", _ @ _*)  => MergeStrategy.discard
  case PathList("org.apache.hadoop", xs @ _*)  => MergeStrategy.first
  case PathList("com.amazonaws", xs @ _*)  => MergeStrategy.last
  case PathList("com.fasterxml.jackson") => MergeStrategy.first
  case PathList("META-INF", "io.netty.versions.properties")  => MergeStrategy.first
  case PathList("org", "tensorflow", _ @ _*)  => MergeStrategy.first
  case x =>
    val oldStrategy = (assemblyMergeStrategy in assembly).value
    oldStrategy(x)
}

// DO NOT CHANGE. 'eval' is a reserved sbt word. Using 'evaluation' for sbt reference
lazy val evaluation = (project in file("eval"))
  .settings(
    name := "spark-nlp-eval",
    version := "2.2.1",

    assemblyMergeStrategy in assembly := evalMergeRules,

    libraryDependencies ++= testDependencies ++ Seq(
      "org.mlflow" % "mlflow-client" % "1.0.0"
    ),

    test in assembly := {},

    publishTo := Some(
      if (isSnapshot.value)
        Opts.resolver.sonatypeSnapshots
      else
        Opts.resolver.sonatypeStaging
    ),

    homepage := Some(url("https://nlp.johnsnowlabs.com")),

    scmInfo := Some(
      ScmInfo(
        url("https://github.com/JohnSnowLabs/spark-nlp"),
        "scm:git@github.com:JohnSnowLabs/spark-nlp.git"
      )
    ),

    credentials += Credentials(Path.userHome / ".ivy2" / ".sbtcredentials"),

    ivyScala := ivyScala.value map {
      _.copy(overrideScalaVersion = true)
    },
    organization := "com.johnsnowlabs.nlp",

    licenses += "Apache-2.0" -> url("http://opensource.org/licenses/Apache-2.0")

  )
  .dependsOn(root)

lazy val ocr = (project in file("ocr"))
  .settings(
    name := "spark-nlp-ocr",
    version := "2.2.1",

    test in assembly := {},

    libraryDependencies ++= ocrDependencies ++
      analyticsDependencies ++
      testDependencies,
    assemblyMergeStrategy in assembly := ocrMergeRules,
    sonatypeProfileName := "com.johnsnowlabs",

    publishTo := Some(
      if (isSnapshot.value)
        Opts.resolver.sonatypeSnapshots
      else
        Opts.resolver.sonatypeStaging
    ),

    homepage := Some(url("https://nlp.johnsnowlabs.com")),

    scmInfo := Some(
      ScmInfo(
        url("https://github.com/JohnSnowLabs/spark-nlp"),
        "scm:git@github.com:JohnSnowLabs/spark-nlp.git"
      )
    ),
    credentials += Credentials(Path.userHome / ".ivy2" / ".sbtcredentials"),

    ivyScala := ivyScala.value map {
      _.copy(overrideScalaVersion = true)
    },
    organization := "com.johnsnowlabs.nlp",

    licenses += "Apache-2.0" -> url("http://opensource.org/licenses/Apache-2.0")
  )
  .dependsOn(root % "test")

parallelExecution in Test := false

logBuffered in Test := false

scalacOptions ++= Seq(
  "-feature",
  "-language:implicitConversions"
)

/** Enable for debugging */
testOptions in Test += Tests.Argument("-oF")

/** Disables tests in assembly */
test in assembly := {}

/** Publish test artificat **/
publishArtifact in Test := true

/** Copies the assembled jar to the pyspark/lib dir **/
lazy val copyAssembledJar = taskKey[Unit]("Copy assembled jar to pyspark/lib")
lazy val copyAssembledOcrJar = taskKey[Unit]("Copy assembled ocr jar to pyspark/lib")
lazy val copyAssembledEvalJar = taskKey[Unit]("Copy assembled eval jar to pyspark/lib")
lazy val copyAssembledJarForPyPi = taskKey[Unit]("Copy assembled jar to pyspark/sparknlp/lib")

copyAssembledJar := {
  val jarFilePath = (assemblyOutputPath in assembly).value
  val newJarFilePath = baseDirectory( _ / "python" / "lib" /  "sparknlp.jar").value
  IO.copyFile(jarFilePath, newJarFilePath)
  println(s"[info] $jarFilePath copied to $newJarFilePath ")
}

copyAssembledOcrJar := {
  val jarFilePath = (assemblyOutputPath in assembly in "ocr").value
  val newJarFilePath = baseDirectory( _ / "python" / "lib" /  "sparknlp-ocr.jar").value
  IO.copyFile(jarFilePath, newJarFilePath)
  println(s"[info] $jarFilePath copied to $newJarFilePath ")
}

// Includes spark-nlp, so use sparknlp.jar
copyAssembledEvalJar := {
  val jarFilePath = (assemblyOutputPath in assembly in "evaluation").value
  val newJarFilePath = baseDirectory( _ / "python" / "lib" /  "sparknlp.jar").value
  IO.copyFile(jarFilePath, newJarFilePath)
  println(s"[info] $jarFilePath copied to $newJarFilePath ")
}


copyAssembledJarForPyPi := {
  val jarFilePath = (assemblyOutputPath in assembly).value
  val newJarFilePath = baseDirectory( _ / "python" / "sparknlp" / "lib"  / "sparknlp.jar").value
  IO.copyFile(jarFilePath, newJarFilePath)
  println(s"[info] $jarFilePath copied to $newJarFilePath ")
}
