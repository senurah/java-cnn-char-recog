# Character Recognition Convolutional Neural Network (CNN) using Java 

This project implements a Convolutional Neural Network (CNN) for character recognition in Java. It was developed to practice and gain an understanding of CNNs.

## How to use
1) Download the MNIST data set using the provided link below.
2) Change the "example_path" paths in the Main.class
   
   ``  List<Image> imagesTest = new DataReader().readData("example_path/mnist_test.csv");``
   
   `` List<Image> imagesTrain = new DataReader().readData("example_path/mnist_train.csv");``

## Dataset
This project uses the MNIST dataset for handwritten digits by default. You can download it from [here](http://yann.lecun.com/exdb/mnist/). Place the dataset in the `data` directory.

## Acknowledgements
This project was inspired by the YouTube video series "Character Recognition with CNN in Java" by [Rae is Online](https://www.youtube.com/channel/raeisonline7254). Special thanks to the channel for the valuable tutorials that helped in understanding CNN concepts.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact
For any questions or suggestions, feel free to open an issue or contact me at [email](neophytetoskill@gmail.com).

