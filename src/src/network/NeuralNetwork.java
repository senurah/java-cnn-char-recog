package network;

import data.Image;
import layers.Layer;

import java.util.ArrayList;
import java.util.List;

import static data.MatrixUtility.add;
import static data.MatrixUtility.multiply;

public class NeuralNetwork {

    List<Layer> _layers;
    //Scaling issue : should scale down initial value of each input down.
    //Using a scale factor
    double scaleFactor;


    public NeuralNetwork(List<Layer> _layers,double scaleFactor) {
        this._layers = _layers;
        this.scaleFactor = scaleFactor;
        linkLayer();

    }

    private void linkLayer(){

        //Checking there is only one layer
        if(_layers.size() <=1){
            return;
        }

        //Going through all the layers
        for(int i = 0; i<_layers.size();i++){
            //for the first layer setting the next layer
            //for last layer: setting the previous layer
            //for middle layer: setting the previous and the next layer

            if(i==0){
                _layers.get(i).set_nextLayer(_layers.get(i+1));
            }else if(i == _layers.size()-1){
                _layers.get(i).set_previousLayer(_layers.get(i-1));
            } else{
                _layers.get(i).set_previousLayer((_layers.get(i-1)));
                _layers.get(i).set_nextLayer(_layers.get(i+1));
            }

        }

    }

    //Getting the errors by checking the corresponding answer and the output
    public double[] getErrors(double[] networkOutput, int correctAnswer){
        //Getting the number of classes
        int numClasses = networkOutput.length;

        double[] expected = new double[numClasses];

        expected[correctAnswer] = 1;

        //Need the actual output - the expected value which gives the error.
        //Multiplying the expected value with -1 : to find the difference between the expected vs actual
        return add(networkOutput,multiply(expected,-1));

    }

    //Figuring out the guess
    private int getMaxIndex(double[] in){

        double max = 0;
        int index = 0;

        for(int i=0; i< in.length;i++){
            if(in[i] >= max){
                max = in[i];
                index = i;
            }
        }

        return index;
    }

    //Guess function
    public int guess(Image image){
        //Extracting the image data and storing it in the inList
        List<double[][]> inList = new ArrayList<>();
        //Before adding to the initial list , scaling down the image by multiplying
        inList.add(multiply(image.getData(),(1.0/scaleFactor)));

        //calling the network to make a guess
        double[] out = _layers.get(0).getOutput(inList);
        int guess = getMaxIndex(out);

        return guess;
    }

    //Test method to use this guess method
    public float test(List<Image> images){

        int correct = 0;

        //Counting the correct guesses
        for(Image image: images){
            int guess = guess(image);

            //Checking the guess is correct
            if(guess == image.getLabel()){
                correct++;
            }
        }
        //Returning the percentage of correct outputs
        //Casting correct to a float
        return ((float) correct/images.size());

    }

    //Train function
    public void train(List<Image> images){

        for(Image img: images){
            //Converting Images to the list format
            List<double[][]> inList = new ArrayList<>();
            //Before adding to the initial list , scaling down the image by multiplying
            inList.add(multiply(img.getData(),(1.0/scaleFactor)));

            double[] out = _layers.get(0).getOutput(inList);
            //Getting the error
            double[] dLdoO = getErrors(out,img.getLabel());

            //Back propagation
            _layers.get(_layers.size()-1).backPropagation(dLdoO);

        }
    }




}
