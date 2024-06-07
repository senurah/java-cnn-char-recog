package layers;

import java.util.ArrayList;
import java.util.List;

public abstract class Layer {

    //Keeping the track of layers - to get the front and back propagation
    protected Layer _nextLayer;
    protected Layer _previousLayer;

    //setting the getters and setters

    public Layer get_nextLayer() {
        return _nextLayer;
    }

    public void set_nextLayer(Layer _nextLayer) {
        this._nextLayer = _nextLayer;
    }

    public Layer get_previousLayer() {
        return _previousLayer;
    }

    public void set_previousLayer(Layer _previousLayer) {
        this._previousLayer = _previousLayer;
    }


    //get output - passing the layer calculations to next layer.

    //The return of this method is the best guess of the model
    //takes vector inputs or matrix list - using polymorphism
    public abstract double[] getOutput(List<double[][]> input);
    public abstract double[] getOutput(double[] input);

    //Taking the loss and the output
    public abstract void backPropagation(double[] dLdO);
    public abstract void backPropagation(List<double[][]> dLdO);



    //To set the dimensions for the next layer using the previous layer
    public abstract  int getOutputLength();
    public abstract int getOutputRows();
    public abstract int getOutputCols();

    //Matrix = length * rows * cols
    public abstract int getOutputElements();


    //converting matrix to vector
    public double[] matrixToVector(List<double[][]> input){

        int length = input.size();
        int rows = input.get(0).length;
        int cols = input.get(0)[0].length;

        double[] vector = new double[length*rows*cols];

        //layering the vector input
        int i = 0;
        //going  through the every matrix in the list of matrices
        for(int l =0; l<length; l++){
            //going through every row
            for (int r =0; r< rows;r++){
                //going through every col
                for(int c =0 ; c< cols; c++){
                    vector[i] = input.get(l)[r][c];
                    i++;
                }
            }
        }

       return vector;
    }

    //converting oneD String into a matrix
    List<double[][]> vectorToMatrix(double[] input ,int length ,int rows ,int cols){

        List<double[][]> out = new ArrayList<>();

        int i = 0;
        for (int l =0; l< length ; l++){

            double [][] matrix = new double[rows][cols];

            for(int r=0; r < rows ; r++){
                for(int c =0 ; c < cols ; c++){
                    matrix[r][c] = input[i];
                    i++;
                }
            }

            //adding matrix in to the List
            out.add(matrix);
        }

        return out;
    }






}
