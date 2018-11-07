package tesis.fraudDetection; /**
 * Created by carlitos on 05/11/18
 */

// $example on$

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.DoubleFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import scala.Tuple2;

import java.util.List;
// $example off$

public class JavaKMeansExample {
    public static void main(String[] args) {

        SparkConf conf = new SparkConf().setAppName("JavaKMeansExample").setMaster("local");
        JavaSparkContext jsc = new JavaSparkContext(conf);

        // $example on$
        // Load and parse data
        String path = "src/main/resources/outputDataTraining.csv";
        JavaRDD<String> data = jsc.textFile(path);
        JavaRDD<Vector> parsedData = data.map(s -> {
            String[] sarray = s.split(",");
            double[] values = new double[sarray.length];
            for (int i = 0; i < sarray.length; i++) {
                values[i] = Double.parseDouble(sarray[i]);
            }
            return Vectors.dense(values);
        });
        parsedData.cache();

        System.out.println("KDD data row size : " + parsedData.count());
        System.out.println("Example data : " + parsedData.first());

        JavaDoubleRDD firstColumn = parsedData.mapToDouble((DoubleFunction<Vector>) t -> {
            // TODO Auto-generated method stub
            return t.apply(0);
        });

        final double mean = firstColumn.mean();
        final double stdev = firstColumn.stdev();

        System.out.println("Meaning value : " + mean + " Standard deviation : " + stdev + " Max : " + firstColumn.max() + " Min : " + firstColumn.min());


        JavaRDD<Vector> filteredParsedDataRDD = parsedData.filter((Function<Vector, Boolean>) v1 -> {
            double src_bytes = v1.apply(0);
            if (src_bytes > (mean - 2 * stdev) && src_bytes < (mean + 2 * stdev)) {
                return true;
            }
            return false;
        }).cache();


        System.out.println("Filtered data ...  Count : " + filteredParsedDataRDD.count());
        System.out.println("Example data : " + filteredParsedDataRDD.first());

        final int numClusters = 10;
        final int numIterations = 20;
        KMeansModel clusters = KMeans.train(filteredParsedDataRDD.rdd(), numClusters, numIterations);


        /**
         * Take cluster centers
         */
        Vector[] clusterCenters = clusters.clusterCenters();

        JavaPairRDD<Double, Vector> result1 = parsedData.mapToPair((PairFunction<Vector, Double, Vector>) point -> {
            int centroidIndex = clusters.predict(point);  //find centroid index
            Vector centroid = clusterCenters[centroidIndex]; //get cluster center (centroid) for given point
            //calculate distance
            double preDis = 0;
            for (int i = 0; i < centroid.size(); i++) {
                preDis = Math.pow((centroid.apply(i) - point.apply(i)), 2);

            }
            double distance = Math.sqrt(preDis);
            return new Tuple2<Double, Vector>(distance, point);
        });

        List<Tuple2<Double, Vector>> result = result1.sortByKey(false).collect();

        //Print top ten points
        for (Tuple2<Double, Vector> tuple : result) {
            System.out.println("Distance " + tuple._1());
        }

        System.out.println("Tamaño: " + result.size());

        jsc.stop();
    }
}