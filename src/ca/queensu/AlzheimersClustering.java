package ca.queensu;

import java.util.List;


import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;

import scala.Tuple2;

public class AlzheimersClustering {
	int numClusters;
	int numIterations;
	Tuple2<JavaRDD<Integer>, Vector[]> results;
	
	
	public AlzheimersClustering(int clustersNum, int iterationsNum, JavaRDD<Vector> mat) {
	    this.numClusters = clustersNum;
	    this.numIterations = iterationsNum;
	    this.results = dataCenters(mat, numClusters, numIterations);
	}
	
	public JavaRDD<Integer> getPredictedResults(){
		return this.results._1();
	}
	
	public Vector[] getClusterCenters() {
		return this.results._2();
	}
	    
	public Tuple2<JavaRDD<Integer>,Vector[]> dataCenters(JavaRDD<Vector> mat, int numClusters, int numIterations) {
		KMeansModel clusters = KMeans.train(mat.rdd(), numClusters, numIterations);
		
		System.out.println("Cluster centers:");
	    Vector[] clusterCenters = clusters.clusterCenters();
	    for (Vector center: clusterCenters) {
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
	    
	    Tuple2<JavaRDD<Integer>,Vector[]> results = 
	            new Tuple2<JavaRDD<Integer>,Vector[]>(predicted_values, clusterCenters);
	    return results;
	}
}
