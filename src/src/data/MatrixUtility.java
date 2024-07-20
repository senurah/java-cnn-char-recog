package data;

public class MatrixUtility {

    //Method to add to matrices
    //2D
    public static  double[][] add(double[][] a, double[][]b){

        double[][] out = new double[a.length][a[0].length];

        for(int i =0; i< a.length; i++){
            for(int j=0; j< a[0].length; j++){
                out[i][j] = a[i][j]+ b[i][j];
            }
        }

        return out;
    }

    //1D
    public static  double[] add(double[] a, double[]b){

        double[] out = new double[a.length];

        for(int i =0; i< a.length; i++){
            out[i] = a[i]+ b[i];
        }

        return out;
    }

    //method to multiply matrices using scalar.
    public static  double[][] multiply(double[][] a, double scalar){

        double[][] out = new double[a.length][a[0].length];

        for(int i =0; i< a.length; i++){
            for(int j=0; j< a[0].length; j++){
                out[i][j] = a[i][j]* scalar;
            }
        }

        return out;
    }

    public static  double[] multiply(double[] a, double scalar){

        double[] out = new double[a.length];

        for(int i =0; i< a.length; i++){
           out[i] = a[i] * scalar;
        }

        return out;
    }


}
