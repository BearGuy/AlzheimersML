package ca.queensu;

import java.util.List;

import org.apache.spark.SparkConf;
//import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;


import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;


public class AlzheimersClustering {

	public static void main(String[] args){
		SparkConf conf = new SparkConf().setAppName("AlzheimersClustering");
	    JavaSparkContext jsc = new JavaSparkContext(conf);
	    
	    JavaRDD<String> csvFile = jsc.textFile("hdfs://bi-hadoop-prod-4157.bi.services.us-south.bluemix.net:8020/tmp/data_transposed.csv");
	    final String header = csvFile.first();
	    JavaRDD<String> csvFileWithoutHeader = csvFile.filter(new Function<String, Boolean>(){
	    	
	    	private static final long serialVersionUID = 1L;
	    	
	    	@Override
            public Boolean call(String s) throws Exception {
                return !s.equalsIgnoreCase(header);
            }
	    });
	    JavaRDD<Vector> mat = csvFileWithoutHeader.map(new Function<String, Vector>(){

			private static final long serialVersionUID = 1L;
			
			@Override
			public Vector call(String arg0) throws Exception {
	            String[] attributes = arg0.split(",");
	            
	            double[] values = new double[attributes.length];
	            for (int i = 0; i < attributes.length; i++) {
	        		values[i] = Double.parseDouble(attributes[i]);
	            }
	            return Vectors.dense(values);  
			}
	    	
	    });
	    mat.cache();
	    
	    int numClusters = 2;
	    int numIterations = 10;
	    KMeansModel clusters = KMeans.train(mat.rdd(), numClusters, numIterations);
	    
	    System.out.println("Cluster centers:");
	    for (Vector center: clusters.clusterCenters()) {
	      System.out.println(" " + center.size());
	    }
	    
	    double cost = clusters.computeCost(mat.rdd());
	    System.out.println("Cost: " + cost);

	    double WSSSE = clusters.computeCost(mat.rdd());
	    System.out.println("Within Set Sum of Squared Errors = " + WSSSE);
	    
	    int[] clustCount = new int[2]; 
	    JavaRDD<Integer> predicted_values = clusters.predict(mat);
	    List<Integer> predictedPoints = predicted_values.toArray();
	    for (Integer predicted: predictedPoints){
	    	if (predicted == 0){
	    		clustCount[0]++;
	    	} else {
	    		clustCount[1]++;
	    	}
	    }
	    System.out.println("Clustering Bin Numbers: " + clustCount[0] + ", " + clustCount[1]);
	    
	    // Save and load model
	    // clusters.save(jsc.sc(), "target/org/apache/spark/JavaKMeansExample/AlzheimersClusteringModel");
	    // KMeansModel sameModel = KMeansModel.load(jsc.sc(),
	    // "target/org/apache/spark/JavaKMeansExample/KMeansModel");
	    
	    
	    jsc.stop();
	    jsc.close();
	}
	
}
