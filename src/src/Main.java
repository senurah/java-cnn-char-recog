import cli.AppConfig;
import cli.CliArgsParser;
import cli.CliParseException;
import cli.ConfigRenderer;
import cli.ConfigValidationException;
import cli.ConfigValidator;
import cli.InteractiveConfigWizard;
import cli.ParameterExplainer;
import data.DataReader;
import data.Image;
import network.ModelSerializer;
import network.NetworkBuilder;
import network.NeuralNetwork;

import java.io.IOException;
import java.util.List;

import static java.util.Collections.shuffle;

public class Main {

    public static void main(String[] args) {
        try {
            execute(args);
        } catch (CliParseException e) {
            System.err.println("[CLI Error] " + e.getMessage());
            System.err.println("Run 'java Main --help' for command-line syntax and options.");
            System.exit(1);
        } catch (ConfigValidationException e) {
            System.err.println("[Configuration Error] " + e.getMessage());
            System.exit(1);
        } catch (Exception e) {
            System.err.println("[Execution Error] " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }

    public static void execute(String[] args) throws CliParseException, ConfigValidationException, IOException {
        CliArgsParser.ParseResult parseResult = CliArgsParser.parse(args);

        // 1. Help flag requested
        if (parseResult.isHelpRequested()) {
            CliArgsParser.printHelp(System.out);
            return;
        }

        AppConfig config;

        // 2. Interactive wizard requested
        if (parseResult.isInteractiveRequested()) {
            InteractiveConfigWizard wizard = new InteractiveConfigWizard(
                    System.in, System.out, parseResult.getConfig().isVerbose());
            config = wizard.runWizard(parseResult.getConfig());
        } else {
            // Flag-based configuration
            config = parseResult.getConfig();
            ConfigValidator.validate(config);
        }

        // 3. Verbose mode outputs educational documentation if enabled without wizard
        if (config.isVerbose() && !config.isInteractive()) {
            ParameterExplainer.printAllDocs(System.out);
            System.out.println();
        }

        // 4. Print ASCII summary table
        ConfigRenderer.print(config);

        // 5. Load datasets
        System.out.println("Starting dataset loading...");
        DataReader reader = new DataReader();
        List<Image> imagesTrain = reader.readData(config.getTrainPath(), config.getTrainLimit());
        List<Image> imagesTest = reader.readData(config.getTestPath(), config.getTestLimit());

        System.out.println("Loaded " + imagesTrain.size() + " training samples.");
        System.out.println("Loaded " + imagesTest.size() + " testing samples.\n");

        if (imagesTrain.isEmpty()) {
            throw new IllegalStateException("Training dataset is empty: " + config.getTrainPath());
        }
        if (imagesTest.isEmpty()) {
            throw new IllegalStateException("Testing dataset is empty: " + config.getTestPath());
        }

        // 6. Build the Convolutional Neural Network
        System.out.println("Constructing neural network layers...");
        NetworkBuilder builder = new NetworkBuilder(
                AppConfig.INPUT_ROWS,
                AppConfig.INPUT_COLS,
                config.getScaleFactor()
        );

        builder.addConvolutionLayer(
                config.getNumFilters(),
                config.getFilterSize(),
                config.getConvStepSize(),
                config.getConvLearningRate(),
                config.getSeed()
        );

        builder.addMaxPoolLayer(
                config.getPoolWindowSize(),
                config.getPoolStepSize()
        );

        builder.addFullyConnectedLayer(
                config.getNumClasses(),
                config.getFcLearningRate(),
                config.getSeed()
        );

        NeuralNetwork net = builder.build();
        System.out.println("Neural network assembled successfully.\n");

        // Phase 2: Load pre-trained weights if requested
        if (config.getLoadPath() != null && !config.getLoadPath().trim().isEmpty()) {
            System.out.println("Loading pre-trained model weights from: " + config.getLoadPath());
            ModelSerializer.loadWeights(net, config.getLoadPath());
            System.out.println("Pre-trained model weights loaded successfully.\n");
        }

        // 7. Initial evaluation before training (or after loading weights)
        float rate;
        if (config.getLoadPath() != null && !config.getLoadPath().trim().isEmpty()) {
            System.out.println("Evaluating model accuracy on test dataset...");
            rate = net.test(imagesTest);
            System.out.printf("Loaded model test accuracy: %.2f%%\n\n", rate * 100.0f);
        } else {
            System.out.println("Evaluating pre-training baseline accuracy...");
            rate = net.test(imagesTest);
            System.out.printf("Pre-training test accuracy: %.2f%%\n\n", rate * 100.0f);
        }

        // 8. Training loop
        int epochs = config.getEpochs();
        if (epochs > 0) {
            System.out.printf("Commencing training for %d epoch(s)...\n", epochs);

            for (int i = 0; i < epochs; i++) {
                System.out.printf("--- Epoch %d/%d ---\n", (i + 1), epochs);
                shuffle(imagesTrain);
                net.train(imagesTrain);
                rate = net.test(imagesTest);
                System.out.printf("Test accuracy after epoch %d: %.2f%%\n\n", (i + 1), rate * 100.0f);
            }
            System.out.println("Training completed successfully.");
        } else {
            System.out.println("Evaluation mode (--epochs 0): training skipped.");
        }

        // Phase 2: Save trained weights if requested
        if (config.getSavePath() != null && !config.getSavePath().trim().isEmpty()) {
            System.out.println("Saving model weights to: " + config.getSavePath());
            ModelSerializer.save(net, config.getSavePath());
            System.out.println("Model weights successfully saved to: " + config.getSavePath() + "\n");
        }
    }
}
