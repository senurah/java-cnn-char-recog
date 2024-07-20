package network;

import layers.ConvolutionLayer;
import layers.FullyConnectedLayer;
import layers.Layer;
import layers.MaxPoolLayer;

import java.util.ArrayList;
import java.util.List;

public class NetworkBuilder {
    //Class to simplify the network process
    private NeuralNetwork net;
    private int _inputRows;
    private int _inputCols;
    //Giving a scale factor
    private double _scaleFactor;
    List<Layer> _layers;

    public NetworkBuilder(int _inputRows, int _inputCols,double _scaleFactor) {
        this._inputRows = _inputRows;
        this._inputCols = _inputCols;
        this._scaleFactor = _scaleFactor;
        //Initializing the layers
        _layers = new ArrayList<>();
    }

    //Method for adding each layer types
    //All the parameters to build the convolution layer
    public void addConvolutionLayer(int numFilters,int filterSize,int stepSize, double learningRate, long SEED){

        //Checking if the layer is the first layer
        if(_layers.isEmpty()){
            //Making the first layer
            _layers.add(new ConvolutionLayer(filterSize,stepSize,1,_inputRows,_inputCols,SEED,numFilters,learningRate));
        } else{
            //Getting the dimensions from the previous layer
            Layer prev = _layers.get(_layers.size()-1);
            _layers.add(new ConvolutionLayer(filterSize,stepSize,prev.getOutputLength(), prev.getOutputRows(),prev.getOutputCols(),SEED,numFilters,learningRate));

        }
    }

    //For the max pool layer
    public void addMaxPoolLayer(int windowSize,int stepSize){
        //Checking if the layer is the first layer
        if(_layers.isEmpty()){
            //Making the first layer
            _layers.add(new MaxPoolLayer(stepSize,windowSize,1,_inputRows,_inputCols));
        } else{
            //Getting the dimensions from the previous layer
            Layer prev = _layers.get(_layers.size()-1);
            _layers.add(new MaxPoolLayer(stepSize,windowSize,prev.getOutputLength(), prev.getOutputRows(),prev.getOutputCols()));

        }
    }

    //For the fully connected layer
    public void addFullyConnectedLayer(int outLength,double learningRate,long SEED){
        if(_layers.isEmpty()){
            _layers.add(new FullyConnectedLayer(_inputCols*_inputRows,outLength,SEED,learningRate));
        } else {
            Layer prev = _layers.get(_layers.size()-1);
            _layers.add(new FullyConnectedLayer(prev.getOutputElements(),outLength,SEED,learningRate));

        }
    }

    //Building the neural network
    public NeuralNetwork build(){
        net = new NeuralNetwork(_layers,_scaleFactor);
        return net;
    }


}
