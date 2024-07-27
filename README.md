1. **Preprocess the dataset:**
    ```sh
    java -cp target/java-cnn-char-recog-1.0-SNAPSHOT.jar com.example.PreprocessDataset
    ```

2. **Train the model:**
    ```sh
    java -cp target/java-cnn-char-recog-1.0-SNAPSHOT.jar com.example.TrainModel
    ```

3. **Evaluate the model:**
    ```sh
    java -cp target/java-cnn-char-recog-1.0-SNAPSHOT.jar com.example.EvaluateModel
    ```

## Dataset
This project uses the MNIST dataset for handwritten digits by default. You can download it from [here](http://yann.lecun.com/exdb/mnist/). Place the dataset in the `data` directory.

## Training
To train the CNN model, use the provided training script. Adjust hyperparameters in the `config.properties` file as needed.

## Evaluation
Evaluate the trained model using the evaluation script. This will provide metrics such as accuracy, precision, and recall.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact
For any questions or suggestions, feel free to open an issue or contact me at [neophytetoskill@gmail.com].
