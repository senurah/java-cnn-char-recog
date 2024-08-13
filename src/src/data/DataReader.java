package data;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

public class DataReader {

    //Creating a class to get the data from the mnist data folder to the image class
    //Converting the double array into an image

    //to track the image size
    private final int rows = 28;
    private final int cols = 28;

    //method to return a list of images
    public List<Image> readData(String path){

        //creating the empty list of images
        List<Image> images = new ArrayList<>();

        /* Reading files using BufferedReader.
         * BufferedReader can read characters efficiently.
         * BufferedReader is used to read the text from a character-based input stream.
         * Internal buffer 8192 characters
         * Reduced number of communications to the disk. --> Efficient
         * Using FileReader class as we can't give the direct file path
         */

        try(BufferedReader dataReader = new BufferedReader(new FileReader(path))){

            String line;

            //looping the lines
            while((line = dataReader.readLine()) != null){
                //Should split the data by "," to get the data values
                String[] lineItems = line.split(",");

                //Converting data into double form
                double[][] data = new double[rows][cols];
                /*
                  In the data set label represent the digit and if we convert that line in to
                  28*28 line we can represent it as a picture.
                */
                //Extracting the label
                int label = Integer.parseInt(lineItems[0]);
                int i = 1;
                for(int row = 0; row < rows; row++){
                    for(int col = 0; col<cols; col++){
                        //Passing and casting to a double
                        data[row][col] = (double) Integer.parseInt(lineItems[i]);
                        i++;
                    }
                }

                //After making it to a 28*28, adding to the Images table(array List)
                images.add(new Image(data,label));

            }

        }catch (Exception e){
            throw new IllegalArgumentException("File not found " + path);
        }
        return images;
    }
}
