/**
 * This is a test for Machine learning linear regression prediction using
 * the WEKA library
 *
 * @author Arthur Buliva
 */
package linearregression.weka;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.core.Instance;
import weka.core.Instances;
import weka.classifiers.functions.LinearRegression;

public class Predictor {

    public static void main(String args[]) {

        Instances data;
        try {
            // Load learning data
            data = new Instances(new BufferedReader(new FileReader("house.arff")));

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
        catch (IOException ex) {
            Logger.getLogger(Predictor.class.getName()).log(Level.SEVERE, null, ex);
        }
        catch (Exception ex) {
            Logger.getLogger(Predictor.class.getName()).log(Level.SEVERE, null, ex);
        }

    }
}
