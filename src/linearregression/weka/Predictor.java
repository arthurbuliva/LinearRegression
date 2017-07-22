/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package linearregression.weka;

/**
 *
 * @author Arthur Buliva
 */
import java.io.BufferedReader;
import java.io.FileReader;
import weka.core.Instance;
import weka.core.Instances;
import weka.classifiers.functions.LinearRegression;

public class Predictor {

    public static void main(String args[]) throws Exception {
        // Load learning data
        Instances data = new Instances(new BufferedReader(new FileReader("house.arff")));
        
        // Remove the last instance that we want to predict
        data.setClassIndex(data.numAttributes() - 1);
        
        // Build the model
        LinearRegression model = new LinearRegression();
        model.buildClassifier(data);
        
        // System.out.println(model);
        
        Instance myHouse = data.lastInstance();
        double price = model.classifyInstance(myHouse);
        
        System.out.println("Expected price is 425900");
        
        System.out.println("My house (" + myHouse + "): " + price);
    }
}
