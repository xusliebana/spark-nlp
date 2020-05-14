import unittest
import shutil
import tempfile

from sparknlp.common import RegexRule
from sparknlp.util import *

from sparknlp.base import *
from sparknlp.annotator import *


class UtilitiesTestSpec(unittest.TestCase):

    @staticmethod
    def runTest():
        regex_rule = RegexRule("\w+", "word split")
        assert(regex_rule.rule() == "\w+")


class SerializersTestSpec(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def serialize_them(self, cls, dirname):
        f = self.test_dir + dirname
        c1 = cls()
        c1.save(f)
        c2 = cls().load(f)
        assert(c1.uid == c2.uid)

    def runTest(self):
        self.serialize_them(DocumentAssembler, "assembler")
        self.serialize_them(TokenAssembler, "token_assembler")
        self.serialize_them(Finisher, "finisher")
        self.serialize_them(Tokenizer, "tokenizer")
        self.serialize_them(Stemmer, "stemmer")
        self.serialize_them(Normalizer, "normalizer")
        self.serialize_them(RegexMatcher, "regex_matcher")
        self.serialize_them(Lemmatizer, "lemmatizer")
        self.serialize_them(DateMatcher, "date_matcher")
        self.serialize_them(TextMatcher, "entity_extractor")
        self.serialize_them(PerceptronApproach, "perceptron_approach")
        self.serialize_them(SentenceDetector, "sentence_detector")
        self.serialize_them(SentimentDetector, "sentiment_detector")
        self.serialize_them(ViveknSentimentApproach, "vivekn")
        self.serialize_them(NorvigSweetingApproach, "norvig")
        self.serialize_them(NerCrfApproach, "ner_crf")


class NerEvaluatorTestSpec(unittest.TestCase):
    def setUp(self):
        self.session = SparkContextForTest.spark
        self.data = SparkContextForTest.data
    def runTest(self):
        ground_truth = ["B-Location", "B-Age", "B-TestResult", "B-UnspecificTherapy", "B-Age", "O", "B-CancerDx", "O", "O", "O", "B-UnspecificTherapy", "B-UnspecificTherapy", "I-HormonalTherapy", "B-Age", "I-PlanHeader", "I-HormonalTherapy", "B-Location", "B-Location", "I-HormonalTherapy", "B-Age", "B-UnspecificTherapy", "O", "B-Age", "B-CancerDx", "O", "I-PlanHeader", "B-Location", "B-Age", "I-PlanHeader", "B-TestResult", "B-Age", "I-HormonalTherapy", "B-Age", "B-TestResult", "B-UnspecificTherapy", "B-Location", "O", "B-CancerDx", "B-TestResult", "B-CancerDx", "I-PlanHeader", "B-Age", "B-Age", "O", "I-HormonalTherapy", "B-UnspecificTherapy", "B-CancerDx", "B-TestResult", "O", "B-TestResult", "B-CancerDx", "B-Location", "B-Age", "O", "B-TestResult", "B-Age", "I-HormonalTherapy", "B-Location", "B-UnspecificTherapy", "B-CancerDx", "B-UnspecificTherapy", "B-TestResult", "I-HormonalTherapy", "I-PlanHeader", "I-PlanHeader", "B-CancerDx", "I-HormonalTherapy", "I-HormonalTherapy", "B-Location", "I-HormonalTherapy", "B-TestResult", "I-PlanHeader", "B-CancerDx", "I-PlanHeader", "B-Age", "I-PlanHeader", "B-Location", "B-Location", "B-CancerDx", "B-Location", "B-Location", "B-Age", "B-CancerDx", "B-CancerDx", "B-CancerDx", "B-TestResult", "I-HormonalTherapy", "B-TestResult", "B-Location", "I-HormonalTherapy", "B-Age", "O", "B-Location", "B-TestResult", "B-CancerDx", "B-CancerDx", "B-Location", "I-PlanHeader", "I-PlanHeader", "B-Age"]
        predictions = ["I-HormonalTherapy", "B-TestResult", "B-TestResult", "B-Location", "B-Location", "B-CancerDx", "B-CancerDx", "I-HormonalTherapy", "I-HormonalTherapy", "B-Age", "B-Location", "O", "B-TestResult", "I-PlanHeader", "B-Location", "B-TestResult", "B-Location", "B-TestResult", "B-CancerDx", "B-Location", "B-Location", "B-CancerDx", "I-PlanHeader", "B-UnspecificTherapy", "B-UnspecificTherapy", "I-HormonalTherapy", "B-UnspecificTherapy", "I-PlanHeader", "B-CancerDx", "B-UnspecificTherapy", "I-PlanHeader", "B-Location", "B-Age", "B-CancerDx", "B-CancerDx", "B-TestResult", "B-Age", "B-UnspecificTherapy", "I-HormonalTherapy", "I-HormonalTherapy", "I-HormonalTherapy", "O", "I-PlanHeader", "O", "B-Location", "B-Location", "B-TestResult", "B-UnspecificTherapy", "B-CancerDx", "B-CancerDx", "I-HormonalTherapy", "B-UnspecificTherapy", "B-Age", "O", "B-TestResult", "I-PlanHeader", "B-TestResult", "I-PlanHeader", "I-PlanHeader", "I-PlanHeader", "B-TestResult", "B-TestResult", "I-HormonalTherapy", "I-HormonalTherapy", "B-UnspecificTherapy", "I-HormonalTherapy", "I-PlanHeader", "I-HormonalTherapy", "B-Location", "I-HormonalTherapy", "B-Age", "B-CancerDx", "O", "B-UnspecificTherapy", "B-TestResult", "O", "O", "B-TestResult", "I-HormonalTherapy", "O", "B-TestResult", "B-Age", "B-CancerDx", "B-TestResult", "B-Age", "B-UnspecificTherapy", "I-HormonalTherapy", "I-HormonalTherapy", "O", "O", "B-CancerDx", "B-Age", "B-Location", "I-PlanHeader", "I-HormonalTherapy", "B-TestResult", "B-Location", "B-Age", "I-HormonalTherapy", "B-Age"]
        t = NerEvaluator.evaluateNer(self.session, ground_truth, predictions, output_df=True)
        print(type(t))