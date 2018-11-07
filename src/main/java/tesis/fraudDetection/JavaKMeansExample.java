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

import java.sql.Timestamp;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.List;
// $example off$

public class JavaKMeansExample {
    public static void main(String[] args) {

        SparkConf conf = new SparkConf().setAppName("JavaKMeansExample").setMaster("local").set("spark.hadoop.validateOutputSpecs", "false");
        JavaSparkContext jsc = new JavaSparkContext(conf);

        // $example on$
        // Load and parse data
        String path = "datasets/llamadas_062018.csv";
        JavaRDD<String> data = getTransformedData(jsc, path);
        JavaRDD<Vector> parsedData = data.map(s -> {
            String[] sarray = s.split(",");
            double[] values = new double[sarray.length];
            for (int i = 0; i < sarray.length; i++) {
                try{
                    values[i] = Double.parseDouble(sarray[i]);
                }catch(NumberFormatException e ){
                    values[i] = Double.parseDouble(String.valueOf(sarray[i].hashCode()));

                }
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


        JavaRDD<Vector>[] filteredParsedDataRDD = parsedData.filter((Function<Vector, Boolean>) v1 -> {
            double src_bytes = v1.apply(0);
            if (src_bytes > (mean - 2 * stdev) && src_bytes < (mean + 2 * stdev)) {
                return true;
            }
            return false;
        }).cache().randomSplit(new double[]{0.8, 0.2});
        JavaRDD<Vector> training = filteredParsedDataRDD[0]; // training set
        JavaRDD<Vector> test = filteredParsedDataRDD[1]; // test set

        System.out.println("Filtered data ...  Count : " + training.count());
        System.out.println("Example data : " + training.first());

        final int numClusters = 10;
        final int numIterations = 20;
        KMeansModel clusters = KMeans.train(training.rdd(), numClusters, numIterations);


        /**
         * Take cluster centers
         */
        Vector[] clusterCenters = clusters.clusterCenters();
        int[] clusterCounters = new int[numClusters];
        Arrays.fill(clusterCounters, 0);

        JavaPairRDD<Double, Integer> result1 = training.mapToPair((PairFunction<Vector, Double, Integer>) point -> {
            int centroidIndex = clusters.predict(point);  //find centroid index
            Vector centroid = clusterCenters[centroidIndex]; //get cluster center (centroid) for given point
            //calculate distance
            double preDis = 0;
            for (int i = 0; i < centroid.size(); i++) {
                preDis = Math.pow((centroid.apply(i) - point.apply(i)), 2);

            }
            double distance = Math.sqrt(preDis);
            return new Tuple2<Double, Integer>(distance, centroidIndex);
        });

        List<Tuple2<Double, Integer>> result = result1.sortByKey(false).collect();

        //Print top ten points
        ArrayList<String> results = new ArrayList<>();

        for (Tuple2<Double, Integer> tuple : result) {
            results.add("Distance " + tuple._1());
            clusterCounters[tuple._2()]++;
        }
        results.add("Tamaño de entrenamiento: " + training.rdd().count());

        results.add("Tamaño de prueba: " + result.size());

        results.add("Contadores de clusteres" );
        int i=0;
        for (int counter : clusterCounters){
            results.add("Cluster: " + i + ":  " + counter );
            i++;
        }



        JavaRDD<String> resultsRdd = jsc.parallelize(results);

        resultsRdd.coalesce(1).saveAsTextFile("finalResults");

        jsc.stop();
    }

    private static JavaRDD<String> getTransformedData(JavaSparkContext jsc, String path) {
        int[] columsTypeString = {2, 4, 11, 12, 13, 15, 20, 21};
        int[] columsTypeDate = {5, 8, 9};

        JavaRDD<String> inputData = jsc.textFile(path);

        JavaRDD<String> inputParseado2 = inputData.filter(line -> line.split(",").length == 22).map( line -> {
            String[] parts = line.split(",");
            StringBuilder result = new StringBuilder();
            for (int i: columsTypeString){
                if(result.length() > 0)
                    result.append(",");
                result.append(parts[i]);
            }
            return result.toString();
        });
        ArrayList<List<String>> replacements = new ArrayList<>();
        for (int j=0; j<columsTypeString.length; j++){
            int finalJ = j;
            JavaRDD<String> rddTemp = inputParseado2.map(line -> {
                String[] temp = line.split(",");
                return temp[finalJ];
            });
            JavaRDD<String> rddDistinct = rddTemp.distinct();
            replacements.add(rddDistinct.collect());
        }
        return inputData.filter(line -> line.split(",").length == 22).map(line -> {
            for (List<String> list: replacements){
                Integer i = 0;
                for(String dato: list){
                    line = line.replaceAll("^"+ dato + ",", i.toString() + ",")
                            .replaceAll(","+ dato + ",", "," + i.toString() + ",")
                            .replaceAll(","+ dato + "$", "," + i.toString());
                    i++;
                }
            }
            String[] parts = line.split(",");
            StringBuilder result = new StringBuilder();
            SimpleDateFormat formatDate = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS");
            //Convertir las fechas a su equivalente en timestamp (long)
            try{
                for (int i : columsTypeDate) {
                    Date date = formatDate.parse(parts[i]);
                    Timestamp timestamp = new Timestamp(date.getTime());
                    parts[i] = String.valueOf(timestamp.getTime());
                }
            }catch(Exception e){
                System.out.println(line);
            }
            result.append(String.join(",", parts));

            return result.toString();
        });
    }
}