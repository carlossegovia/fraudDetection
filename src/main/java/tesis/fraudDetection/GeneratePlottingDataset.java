package tesis.fraudDetection;

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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class GeneratePlottingDataset {

    public static void main(String[] args) {

        SparkConf conf = new SparkConf().setAppName("GeneratePlotting").setMaster("local").set("spark.hadoop.validateOutputSpecs", "false");
        JavaSparkContext jsc = new JavaSparkContext(conf);

        // $example on$
        // Load and parse data
        String path = "datasets/part-00000";
        // data contiene los datos parseados correctamente.
        JavaRDD<String> data = jsc.textFile(path).cache();

        // Se recorre el rdd y se transforma en un vector de doubles para poder ejecutar el kmeans
        JavaRDD<Vector> parsedData = data.map(s -> {
            String[] sarray = s.split(",");
            double[] values = new double[sarray.length];
            for (int i = 0; i < sarray.length; i++) {
                try {
                    values[i] = Double.parseDouble(sarray[i]);
                } catch (NumberFormatException e) {
                    // En el caso de que sea un string genera un hash y parsea el resultado
                    // Se puede dar este caso porque en la columna de líneas por un problema del dataset original
                    // existen campos como 09082313B0902341 que tienen una "B" en medio
                    values[i] = Double.parseDouble(String.valueOf(sarray[i].hashCode()));
                }
            }
            return Vectors.dense(values);
        }).cache();

        // Se libera de la memoria el rdd data.
        data.unpersist();

        System.out.println("Dataset row size : " + parsedData.count());
        System.out.println("Example data : " + parsedData.first());

        // firstColumn se utiliza para calcular la desviación estandar
        JavaDoubleRDD firstColumn = parsedData.mapToDouble((DoubleFunction<Vector>) t -> {
            // TODO Auto-generated method stub
            return t.apply(0);
        }).cache();

        final double mean = firstColumn.mean();
        final double stdev = firstColumn.stdev();

        System.out.println("Meaning value : " + mean + " Standard deviation : " + stdev + " Max : " + firstColumn.max() + " Min : " + firstColumn.min());

        // Se libera de la memoria el rdd firstColumn
        firstColumn.unpersist();

        // Se normaliza el dataset
        JavaRDD<Vector>[] filteredParsedDataRDD = parsedData.filter((Function<Vector, Boolean>) v1 -> {
            double src_bytes = v1.apply(0);
            if (src_bytes > (mean - 2 * stdev) && src_bytes < (mean + 2 * stdev)) {
                return true;
            }
            return false;
        }).cache().randomSplit(new double[]{0.8, 0.2});
        JavaRDD<Vector> training = filteredParsedDataRDD[0]; // training set
        JavaRDD<Vector> test = filteredParsedDataRDD[1]; // test set (no se usa)

        // Se libera de la memoria ambos rdds
        filteredParsedDataRDD[0].unpersist();
        filteredParsedDataRDD[1].unpersist();

        training.cache();
        System.out.println("Filtered data ...  Count : " + training.count());
        System.out.println("Example data : " + training.first());

        //Se ejecuta kmeans
        int numClusters = 10;
        final int numIterations = 10;
        KMeansModel clusters = KMeans.train(training.rdd(), numClusters, numIterations);


        /**
         * Take cluster centers
         */
        int[] clusterCounters = new int[numClusters];
        Arrays.fill(clusterCounters, 0);

        JavaRDD<String> result1 = test.map(point -> {
            Integer centroidIndex = clusters.predict(point);  //find centroid index
            ArrayList<String> list = new ArrayList<>();
            list.add(centroidIndex.toString());
            for (Double e : point.toArray()){
                list.add(e.toString());
            }
            //calculate distance
            return String.join(",", list);
        }).cache();

        List<String> result = result1.takeSample(false, 5000, System.currentTimeMillis());

        result1.unpersist();


        // Se guardan los resultados.
        JavaRDD<String> resultsRdd = jsc.parallelize(result).cache();

        resultsRdd.coalesce(1).saveAsTextFile("results/plottingData");

        // liberar de la memoria el rdd training
        resultsRdd.unpersist();


        jsc.stop();
    }
}
