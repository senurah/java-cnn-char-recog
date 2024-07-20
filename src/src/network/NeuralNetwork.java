package network;

import data.Image;
import layers.Layer;

import java.util.ArrayList;
import java.util.List;

import static data.MatrixUtility.add;
import static data.MatrixUtility.multiply;

public class NeuralNetwork {

    List<Layer> _layers;

    public NeuralNetwork(List<Layer> _layers) {
        this._layers = _layers;
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
        inList.add(image.getData());

        return 1;
    }




}
