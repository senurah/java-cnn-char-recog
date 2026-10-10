package data;
import java.io.BufferedReader;
import java.io.File;
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
        return readData(path, 0);
    }

    public List<Image> readData(String path, int limit){

        //creating the empty list of images
        List<Image> images = new ArrayList<>();

        // Handle possible path resolution from project root or subdirectories
        File file = new File(path);
        if(!file.exists()){
            if(new File("data/" + file.getName()).exists()){
                file = new File("data/" + file.getName());
            } else if(new File("../../data/" + file.getName()).exists()){
                file = new File("../../data/" + file.getName());
            }
        }

        try(BufferedReader dataReader = new BufferedReader(new FileReader(file))){

            String line;

            //looping the lines
            while((line = dataReader.readLine()) != null){
                if(limit > 0 && images.size() >= limit){
                    break;
                }
                if(line.trim().isEmpty()){
                    continue;
                }

                //Should split the data by "," to get the data values
                String[] lineItems = line.split(",");

                // Skip header row if present
                if(lineItems[0].equalsIgnoreCase("label")){
                    continue;
                }

                //Converting data into double form
                double[][] data = new double[rows][cols];
                /*
                  In the data set label represent the digit and if we convert that line in to
                  28*28 line we can represent it as a picture.
                */
                //Extracting the label
                int label = Integer.parseInt(lineItems[0].trim());
                int i = 1;
                for(int row = 0; row < rows; row++){
                    for(int col = 0; col<cols; col++){
                        //Passing and casting to a double
                        data[row][col] = (double) Integer.parseInt(lineItems[i].trim());
                        i++;
                    }
                }

                //After making it to a 28*28, adding to the Images table(array List)
                images.add(new Image(data,label));

            }

        }catch (Exception e){
            System.err.println("Error reading dataset file: " + file.getAbsolutePath());
            e.printStackTrace();
            throw new IllegalArgumentException("Could not load data from " + path + ": " + e.getMessage(), e);
        }
        return images;
    }
}
