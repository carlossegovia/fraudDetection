package tesis.fraudDetection;

/**
 * Created by carlitos on 08/09/18
 */

import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.SparkConf;
import org.spark_project.dmg.pmml.ConfusionMatrix;
import scala.Tuple2;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.stream.Stream;

public class NaiveBayesExample {

    public static void main(String[] args) {
        SparkConf sparkConf = new SparkConf().setAppName("JavaNaiveBayesExample").setMaster("local[2]");
        JavaSparkContext jsc = new JavaSparkContext(sparkConf);
        String path = null;
        if (args != null && args.length > 0) {
            path = args[0];
        } else {
            path = "datasets/kddcup_normalizado.data";
        }
//        JavaRDD<LabeledPoint> inputData = MLUtils.loadLibSVMFile(jsc.sc(), path).toJavaRDD();
        JavaRDD<LabeledPoint> inputData = null;
        try {
            inputData = loadDataFromFileKDD(jsc, path);

            JavaRDD<LabeledPoint>[] tmp = inputData.randomSplit(new double[]{0.8, 0.2});
            JavaRDD<LabeledPoint> training = tmp[0]; // training set
            JavaRDD<LabeledPoint> test = tmp[1]; // test set
            NaiveBayesModel model = NaiveBayes.train(training.rdd(), 1.0);
            JavaPairRDD<Object, Object> predictionAndLabel =
                    test.mapToPair(p -> new Tuple2<>(model.predict(p.features()), p.label()));

            BinaryClassificationMetrics metrics = new BinaryClassificationMetrics(predictionAndLabel.rdd());
            MulticlassMetrics metrics2 = new MulticlassMetrics(predictionAndLabel.rdd());

            // Save and load model
            model.save(jsc.sc(), "target/tmp/myNaiveBayesModel");
            NaiveBayesModel sameModel = NaiveBayesModel.load(jsc.sc(), "target/tmp/myNaiveBayesModel");
            // $example off$
            JavaRDD<Tuple2<Object, Object>> precision = metrics.precisionByThreshold().toJavaRDD();
            System.out.println("Precision by threshold: " + precision.collect());

            // Recall by threshold
            JavaRDD<?> recall = metrics.recallByThreshold().toJavaRDD();
            System.out.println("Recall by threshold: " + recall.collect());


            Matrix confusion = metrics2.confusionMatrix();
            System.out.println("Confusion matrix: \n" + confusion);


            // Overall statistics
            System.out.println("Accuracy = " + metrics2.accuracy());
            jsc.stop();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    static JavaRDD<LabeledPoint> loadDataFromFile(JavaSparkContext sc, String path) throws IOException {
        File file = new File(path);

        return sc.textFile(file.getPath()).
                map(line -> {
                    line = line.replace("PAYMENT", "1")
                            .replace("TRANSFER", "2")
                            .replace("CASH_OUT", "3")
                            .replace("DEBIT", "4")
                            .replace("CASH_IN", "5")
                            .replace("C", "1")
                            .replace("M", "2");
                    String[] split = line.split(",");
                    //skip header
                    if (split[0].equalsIgnoreCase("step")) {
                        return null;
                    }
                    double[] featureValues = Stream.of(split)
                            .mapToDouble(e -> Double.parseDouble(e)).toArray();

                    //always skip 9 and 10 because they are labels fraud or not fraud
                    double label = featureValues[9];
                    featureValues = Arrays.copyOfRange(featureValues, 0, 9);
                    return new LabeledPoint(label, Vectors.dense(featureValues));
                });
    }


    static JavaRDD<LabeledPoint> loadDataFromFileKDD(JavaSparkContext sc, String path) throws IOException {
        File file = new File(path);

        return sc.textFile(file.getPath()).
                map(line -> {
                    String[] split = line.split(",");
                    //skip header
                    if (split[0].equalsIgnoreCase("step")) {
                        return null;
                    }
                    double[] featureValues = Stream.of(split)
                            .mapToDouble(e -> Double.parseDouble(e)).toArray();

                    //always skip 9 and 10 because they are labels fraud or not fraud
                    double label = featureValues[41] != 4.0 ? 1.0 : 0.0;
                    featureValues = Arrays.copyOfRange(featureValues, 0, 41);
                    return new LabeledPoint(label, Vectors.dense(featureValues));
                });
    }
}
