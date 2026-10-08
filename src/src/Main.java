import data.DataReader;
import data.Image;
import network.NetworkBuilder;
import network.NeuralNetwork;

import java.util.List;

import static java.util.Collections.shuffle;

public class Main {
    public static void main(String[] args) {

        long SEED = 123;

        System.out.println("Starting data loading.....");
        List<Image> imagesTest = new DataReader().readData("data/mnist_test.csv");
        List<Image> imagesTrain = new DataReader().readData("data/mnist_train.csv");

        System.out.println("Images Train size: " + imagesTrain.size());
        System.out.println("Images Test size : " + imagesTest.size());

        // Support quick sanity run with "--quick" flag
        boolean quickRun = args.length > 0 && args[0].equalsIgnoreCase("--quick");
        if (quickRun) {
            System.out.println("Running in quick-test mode (200 train / 100 test samples)...");
            imagesTest = imagesTest.subList(0, 100);
            imagesTrain = imagesTrain.subList(0, 200);
        }

        //Building the network
        /*
        convolution layer - 1
        maxPool layer - 1
        fully connected layer - 1
        scale factor - 256*100
        * */
        NetworkBuilder builder = new NetworkBuilder(28,28,256*100);
        builder.addConvolutionLayer(8,5,1,0.1,SEED);
        builder.addMaxPoolLayer(3,2);
        builder.addFullyConnectedLayer(10,0.1,SEED);

        NeuralNetwork net = builder.build();

        //Testing the success rate, before training
        float rate = net.test(imagesTest);
        System.out.println("Pre training success rate: " + rate);

        int epochs = 3;
        for(int i=0; i< epochs; i++){
            //Shuffling the training images
            shuffle(imagesTrain);
            net.train(imagesTrain);
            rate = net.test(imagesTest);
            System.out.println("Success rate after round " + i + ": " + rate);
        }

    }
}
