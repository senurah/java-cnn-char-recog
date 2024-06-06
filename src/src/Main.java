import data.DataReader;
import data.Image;

import java.util.List;

public class Main {
    public static void main(String[] args) {

        //getting the data to the data reader
        List<Image> images = new DataReader().readData("src/data/mnist_test.csv");
        System.out.printf(images.get(0).toString());
    }
}



