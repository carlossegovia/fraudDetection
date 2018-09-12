package tesis.fraudDetection;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import scala.Tuple2;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.apache.spark.mllib.util.MLUtils;

import static tesis.fraudDetection.NaiveBayesExample.loadDataFromFileKDD;

public class RandomForestExample {

    public static void main(String[] args) {
        SparkConf sparkConf = new SparkConf().setAppName("JavaNaiveBayesExample").setMaster("local[2]");
        JavaSparkContext jsc = new JavaSparkContext(sparkConf);
        String path = null;
        if (args != null && args.length > 0) {
            path = args[0];
        } else {
            path = "datasets/kddcup_normalizado.data";
        }
//        JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(jsc.sc(), datapath).toJavaRDD();
        JavaRDD<LabeledPoint> data = null;
        try {
            data = loadDataFromFileKDD(jsc, path);
            // Split the data into training and test sets (30% held out for testing)
            JavaRDD<LabeledPoint>[] splits = data.randomSplit(new double[]{0.7, 0.3});
            JavaRDD<LabeledPoint> trainingData = splits[0];
            JavaRDD<LabeledPoint> testData = splits[1];

            // Train a RandomForest model.
            // Empty categoricalFeaturesInfo indicates all features are continuous.
            Integer numClasses = 2;
            Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
            Integer numTrees = 10; // Use more in practice.
            String featureSubsetStrategy = "auto"; // Let the algorithm choose.
            String impurity = "gini";
            Integer maxDepth = 5;
            Integer maxBins = 32;
            Integer seed = 12345;

//            RandomForestModel model = RandomForest.trainClassifier(trainingData, numClasses,
//                    categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins,
//                    seed);
// Train a DecisionTree model for classification.
            DecisionTreeModel model = DecisionTree.trainClassifier(trainingData, numClasses,
                    categoricalFeaturesInfo, impurity, maxDepth, maxBins);
            // Evaluate model on test instances and compute test error
            JavaPairRDD<Object, Object> predictionAndLabel =
                    testData.mapToPair(p -> new Tuple2<>(model.predict(p.features()), p.label()));

            BinaryClassificationMetrics metrics = new BinaryClassificationMetrics(predictionAndLabel.rdd());
            MulticlassMetrics metrics2 = new MulticlassMetrics(predictionAndLabel.rdd());
            Matrix confusion = metrics2.confusionMatrix();
            System.out.println("Confusion matrix: \n" + confusion);


            // Overall statistics
            System.out.println("Accuracy = " + metrics2.accuracy());

            double precision =  metrics2.precision(metrics2.labels()[0]);
            double recall = metrics2.recall(metrics2.labels()[0]);
            double f_measure = metrics2.fMeasure();
            double WTP = metrics2.weightedTruePositiveRate();
            double WFP =  metrics2.weightedFalsePositiveRate();
            System.out.println("Precision = " + precision);
            System.out.println("Recall = " + recall);
            System.out.println("F-measure = " + f_measure);
            System.out.println("Weighted True Positive Rate = " + WTP);
            System.out.println("Weighted False Positive Rate = " + WFP);

// Save and load model
            model.save(jsc.sc(), "target/tmp/myRandomForestClassificationModel");
            RandomForestModel sameModel = RandomForestModel.load(jsc.sc(),
                    "target/tmp/myRandomForestClassificationModel");
        }catch (IOException e) {
            e.printStackTrace();
        }

    }
}
