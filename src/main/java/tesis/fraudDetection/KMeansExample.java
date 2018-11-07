package tesis.fraudDetection;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import scala.Tuple2;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.stream.Stream;

public class KMeansExample {

    public static void main(String[] args) {
        SparkConf sparkConf = new SparkConf().setAppName("JavaKMeansExample").setMaster("local[2]");
        JavaSparkContext jsc = new JavaSparkContext(sparkConf);
        String path = null;
        if (args != null && args.length > 0) {
            path = args[0];
        } else {
            path = "datasets/kddcup_normalizado.data";
        }
        //        JavaRDD<LabeledPoint> inputData = MLUtils.loadLibSVMFile(jsc.sc(), path).toJavaRDD();
        JavaRDD<String> data = jsc.textFile(path);
        JavaRDD<Vector> parsedData = data.map(s -> {
            String[] sarray = s.split(",");
            double[] values = new double[sarray.length];
            for (int i = 0; i < sarray.length-1; i++) {
                values[i] = Double.parseDouble(sarray[i]);
            }
            return Vectors.dense(values);
        });
        parsedData.cache();

        // Cluster the data into two classes using KMeans
        int numClusters = 2;
        int numIterations = 20;
        KMeansModel clusters = KMeans.train(parsedData.rdd(), numClusters, numIterations);

        System.out.println("Cluster centers:");
        for (Vector center: clusters.clusterCenters()) {
            System.out.println(" " + center);
        }
        double cost = clusters.computeCost(parsedData.rdd());
        System.out.println("Cost: " + cost);

        // Evaluate clustering by computing Within Set Sum of Squared Errors
        double WSSSE = clusters.computeCost(parsedData.rdd());
        System.out.println("Within Set Sum of Squared Errors = " + WSSSE);

        // Save and load model
        clusters.save(jsc.sc(), "target/org/apache/spark/JavaKMeansExample/KMeansModel");
        KMeansModel sameModel = KMeansModel.load(jsc.sc(),
                "target/org/apache/spark/JavaKMeansExample/KMeansModel");

        jsc.stop();
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
