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
        String path = "src/main/resources/inputData.csv";
        // data contiene los datos parseados correctamente.
        JavaRDD<String> data = getTransformedData(jsc, path).cache();

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
        // JavaRDD<Vector> test = filteredParsedDataRDD[1]; // test set (no se usa)

        // Se libera de la memoria ambos rdds
        filteredParsedDataRDD[0].unpersist();
        filteredParsedDataRDD[1].unpersist();

        training.cache();
        System.out.println("Filtered data ...  Count : " + training.count());
        System.out.println("Example data : " + training.first());

        //Se ejecuta kmeans
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
        }).cache();

        List<Tuple2<Double, Integer>> result = result1.sortByKey(false).collect();

        result1.unpersist();

        //Imprimir las distancias
        ArrayList<String> results = new ArrayList<>();
        for (Tuple2<Double, Integer> tuple : result) {
            results.add("Distance " + tuple._1());
            clusterCounters[tuple._2()]++;
        }

        results.add("Tamaño de entrenamiento: " + training.rdd().count());
        results.add("Tamaño de prueba: " + result.size());
        results.add("Contadores de clusteres");

        // liberar de la memoria el rdd training
        training.unpersist();

        //Imprimir la cantidad de datos en cada clusters
        int i = 0;
        for (int counter : clusterCounters) {
            results.add("Cluster: " + i + ":  " + counter);
            i++;
        }

        // Se guardan los resultados.
        JavaRDD<String> resultsRdd = jsc.parallelize(results).cache();

        resultsRdd.coalesce(1).saveAsTextFile("finalResults");

        // liberar de la memoria el rdd training
        resultsRdd.unpersist();

        jsc.stop();
    }

    private static JavaRDD<String> getTransformedData(JavaSparkContext jsc, String path) {
        int[] columsTypeString = {2, 4, 11, 12, 13, 15, 20, 21};
        int[] columsTypeDate = {5, 8, 9};

        // Los datos en bruto
        JavaRDD<String> inputData = jsc.textFile(path).cache();

        // Se parten los rdd's y los que no tienen 22 columnas se obvian
        JavaRDD<String> inputStringType = inputData.filter(line -> line.split(",").length == 22).map(line -> {
            String[] parts = line.split(",");
            // result es el string formado por todas las columnas de tipo string en el dataset original
            StringBuilder result = new StringBuilder();
            for (int i : columsTypeString) {
                if (result.length() > 0)
                    result.append(",");
                result.append(parts[i]);
            }
            return result.toString();
        }).cache();

        // Se forma un Array con todos los diferentes strings encontrados que se van a mapear a un número
        ArrayList<List<String>> replacements = new ArrayList<>();
        for (int j = 0; j < columsTypeString.length; j++) {
            int finalJ = j;
            // Se recorre el rdd generado anteriormente que solo contiene las columnas de tipo string
            JavaRDD<String> rddTemp = inputStringType.map(line -> {
                String[] temp = line.split(",");
                return temp[finalJ];
            }).cache();

            // Se libera de la memoria el rdd inputStringType
            inputStringType.unpersist();

            // El rddTemp contiene los strings que no se repiten, eso se devuelve como lista y se almacena en replacements
            replacements.add(rddTemp.distinct().collect());

            // Se libera de la memoria el rdd rddTemp
            rddTemp.unpersist();
        }

        // Se recorre el dataset para sustituir las columnas de tipo string por el id generado anteriormente y
        // también sustituir las columnas de tipo Date. De vuelta se ignoran las filas que no tienen todos los campos (22)
        JavaRDD<String> rddFinal = inputData.filter(line -> line.split(",").length == 22).map(line -> {
            //Se sustituye cada string contenido en replacements con su correspondiente id que es un number
            for (List<String> list : replacements) {
                Integer i = 0;
                for (String dato : list) {
                    line = line.replaceAll("^" + dato + ",", i.toString() + ",")
                            .replaceAll("," + dato + ",", "," + i.toString() + ",")
                            .replaceAll("," + dato + "$", "," + i.toString());
                    i++;
                }
            }
            // line ya es equivalente a una fila sin columnas con tipos de datos de string.
            String[] parts = line.split(",");
            StringBuilder result = new StringBuilder();
            SimpleDateFormat formatDate = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS");
            Date date;
            Timestamp timestamp;
            //Convertir las fechas a su equivalente en timestamp (long)
            try {
                for (int i : columsTypeDate) {
                    date = formatDate.parse(parts[i]);
                    timestamp = new Timestamp(date.getTime());
                    parts[i] = String.valueOf(timestamp.getTime());
                }
            } catch (Exception e) {
                System.out.println(line);
            }
            // result contiene las filas con todos las columnas con tipos de datos númericos.
            result.append(String.join(",", parts));
            return result.toString();
        });
        // Se libera de la memoria el rdd inputData
        inputData.unpersist();

        return rddFinal;
    }
}