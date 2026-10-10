package test;

import cli.AppConfig;
import cli.CliArgsParser;
import cli.CliParseException;
import cli.ConfigValidationException;
import cli.ConfigValidator;
import data.DataReader;
import data.Image;
import layers.ConvolutionLayer;
import layers.FullyConnectedLayer;
import layers.Layer;
import network.ModelSerializer;
import network.NetworkBuilder;
import network.NeuralNetwork;
import service.ModelPersistenceService;

import javax.swing.SwingUtilities;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

public class Phase2Test {

    private static int testsPassed = 0;
    private static int testsFailed = 0;

    public static void main(String[] args) {
        System.out.println("=== Running Phase 2 Test Suite (Model Weights Persistence) ===");

        try {
            testCliPersistenceParsing();
            testConfigValidatorPersistence();
            testModelSerializerDirectRoundTrip();
            testTrainedModelSaveAndReloadAccuracy();
            testModelSerializerLoadFromScratch();
            testDimensionalMismatchValidation();
            testCorruptedFileAndInvalidHeaderHandling();
            testEventDispatchThreadProtection();
            testAsyncModelPersistenceService();
            testMainExecuteIntegration();
        } catch (Exception e) {
            System.err.println("[CRITICAL TEST ERROR] " + e.getMessage());
            e.printStackTrace();
            testsFailed++;
        }

        System.out.println("\nPhase 2 Test Results: " + testsPassed + " passed, " + testsFailed + " failed.");
        if (testsFailed > 0) {
            System.exit(1);
        }
    }

    private static void assertTrue(String message, boolean condition) {
        if (condition) {
            System.out.println("  [PASS] " + message);
            testsPassed++;
        } else {
            System.err.println("  [FAIL] " + message);
            testsFailed++;
        }
    }

    private static void testCliPersistenceParsing() {
        System.out.println("\n-- Testing TASK-201: CLI Persistence Flags Parsing --");

        try {
            CliArgsParser.ParseResult res = CliArgsParser.parse(new String[]{"--save", "models/net.bin"});
            assertTrue("--save flag parsed", "models/net.bin".equals(res.getConfig().getSavePath()));

            CliArgsParser.ParseResult resLoad = CliArgsParser.parse(new String[]{"--load", "models/net.bin"});
            assertTrue("--load flag parsed", "models/net.bin".equals(resLoad.getConfig().getLoadPath()));

            CliArgsParser.ParseResult resBoth = CliArgsParser.parse(new String[]{
                    "--save=out/saved.bin",
                    "--load=in/loaded.bin",
                    "--epochs", "0"
            });
            assertTrue("--save= inline parsed", "out/saved.bin".equals(resBoth.getConfig().getSavePath()));
            assertTrue("--load= inline parsed", "in/loaded.bin".equals(resBoth.getConfig().getLoadPath()));
            assertTrue("epochs 0 parsed", resBoth.getConfig().getEpochs() == 0);

            String help = CliArgsParser.getHelpMessage();
            assertTrue("Help contains --save", help.contains("--save <PATH>"));
            assertTrue("Help contains --load", help.contains("--load <PATH>"));
        } catch (CliParseException e) {
            assertTrue("CLI parsing failed: " + e.getMessage(), false);
        }
    }

    private static void testConfigValidatorPersistence() {
        System.out.println("\n-- Testing TASK-201: ConfigValidator Persistence Rules --");

        // 1. Empty save path rejected
        try {
            AppConfig invalid = AppConfig.builder().savePath("   ").build();
            ConfigValidator.validate(invalid);
            assertTrue("Empty save path should fail", false);
        } catch (ConfigValidationException e) {
            assertTrue("Empty save path failed as expected: " + e.getMessage(), true);
        }

        // 2. Empty load path rejected
        try {
            AppConfig invalid = AppConfig.builder().loadPath("").build();
            ConfigValidator.validate(invalid);
            assertTrue("Empty load path should fail", false);
        } catch (ConfigValidationException e) {
            assertTrue("Empty load path failed as expected: " + e.getMessage(), true);
        }

        // 3. Epochs 0 allowed if loadPath is present
        try {
            AppConfig evalMode = AppConfig.builder().loadPath("model.bin").epochs(0).build();
            ConfigValidator.validate(evalMode);
            assertTrue("Epochs 0 allowed with loadPath", true);
        } catch (ConfigValidationException e) {
            assertTrue("Epochs 0 with loadPath failed: " + e.getMessage(), false);
        }

        // 4. Epochs 0 rejected if loadPath is null
        try {
            AppConfig invalid = AppConfig.builder().epochs(0).build();
            ConfigValidator.validate(invalid);
            assertTrue("Epochs 0 without loadPath should fail", false);
        } catch (ConfigValidationException e) {
            assertTrue("Epochs 0 without loadPath failed as expected: " + e.getMessage(), true);
        }
    }

    private static void testModelSerializerDirectRoundTrip() throws IOException {
        System.out.println("\n-- Testing TASK-201: ModelSerializer Direct Stream Round-Trip --");

        NetworkBuilder builder = new NetworkBuilder(AppConfig.INPUT_ROWS, AppConfig.INPUT_COLS, 25600.0);
        builder.addConvolutionLayer(4, 5, 1, 0.1, 42L);
        builder.addMaxPoolLayer(3, 2);
        builder.addFullyConnectedLayer(10, 0.1, 42L);
        NeuralNetwork originalNet = builder.build();

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ModelSerializer.save(originalNet, baos);
        byte[] bytes = baos.toByteArray();

        assertTrue("Serialized byte array non-empty", bytes.length > 0);

        // Deserialization from scratch
        ByteArrayInputStream bais = new ByteArrayInputStream(bytes);
        NeuralNetwork reloadedNet = ModelSerializer.load(bais);

        assertTrue("Reloaded net non-null", reloadedNet != null);
        assertTrue("Reloaded net layer count matches", reloadedNet.getLayers().size() == 3);
        assertTrue("Scale factor preserved", Math.abs(reloadedNet.getScaleFactor() - 25600.0) < 1e-9);

        // Compare weights between original and reloaded
        ConvolutionLayer origConv = (ConvolutionLayer) originalNet.getLayers().get(0);
        ConvolutionLayer reloadConv = (ConvolutionLayer) reloadedNet.getLayers().get(0);
        assertTrue("Conv filter count matches", reloadConv.getNumFilters() == origConv.getNumFilters());

        boolean convWeightsIdentical = true;
        for (int f = 0; f < origConv.getNumFilters(); f++) {
            double[][] fOrig = origConv.getFilters().get(f);
            double[][] fReload = reloadConv.getFilters().get(f);
            for (int r = 0; r < origConv.getFilterSize(); r++) {
                for (int c = 0; c < origConv.getFilterSize(); c++) {
                    if (Double.compare(fOrig[r][c], fReload[r][c]) != 0) {
                        convWeightsIdentical = false;
                    }
                }
            }
        }
        assertTrue("Conv filter weights bit-identical", convWeightsIdentical);

        FullyConnectedLayer origFc = (FullyConnectedLayer) originalNet.getLayers().get(2);
        FullyConnectedLayer reloadFc = (FullyConnectedLayer) reloadedNet.getLayers().get(2);

        boolean fcWeightsIdentical = true;
        double[][] wOrig = origFc.getWeights();
        double[][] wReload = reloadFc.getWeights();
        for (int r = 0; r < origFc.getInLength(); r++) {
            for (int c = 0; c < origFc.getOutLength(); c++) {
                if (Double.compare(wOrig[r][c], wReload[r][c]) != 0) {
                    fcWeightsIdentical = false;
                }
            }
        }
        assertTrue("Fully connected weights bit-identical", fcWeightsIdentical);
    }

    private static void testTrainedModelSaveAndReloadAccuracy() throws IOException {
        System.out.println("\n-- Testing TASK-201: Train 1 Epoch, Save, Reload into Fresh Network, Verify Identical Outputs --");

        // Load a slice of real data
        DataReader reader = new DataReader();
        List<Image> trainSamples = reader.readData("data/mnist_train.csv", 60);
        List<Image> testSamples = reader.readData("data/mnist_test.csv", 30);

        assertTrue("Loaded 60 train samples", trainSamples.size() == 60);
        assertTrue("Loaded 30 test samples", testSamples.size() == 30);

        // 1. Build and train Network A
        NetworkBuilder builderA = new NetworkBuilder(AppConfig.INPUT_ROWS, AppConfig.INPUT_COLS, 25600.0);
        builderA.addConvolutionLayer(4, 5, 1, 0.1, 100L);
        builderA.addMaxPoolLayer(3, 2);
        builderA.addFullyConnectedLayer(10, 0.1, 100L);
        NeuralNetwork netA = builderA.build();

        netA.train(trainSamples);
        float accuracyA = netA.test(testSamples);

        // Record individual outputs for all test samples
        List<double[]> outputsA = new ArrayList<>();
        List<Integer> guessesA = new ArrayList<>();
        for (Image img : testSamples) {
            outputsA.add(netA.getOutput(img));
            guessesA.add(netA.guess(img));
        }

        // 2. Save weights of Network A to temporary file
        File tempFile = File.createTempFile("cnn_test_model_", ".bin");
        tempFile.deleteOnExit();

        ModelSerializer.save(netA, tempFile);
        assertTrue("Model weights file written to disk (" + tempFile.length() + " bytes)", tempFile.exists() && tempFile.length() > 0);

        // 3. Build a fresh Network B with DIFFERENT random seed
        NetworkBuilder builderB = new NetworkBuilder(AppConfig.INPUT_ROWS, AppConfig.INPUT_COLS, 25600.0);
        builderB.addConvolutionLayer(4, 5, 1, 0.1, 9999L);
        builderB.addMaxPoolLayer(3, 2);
        builderB.addFullyConnectedLayer(10, 0.1, 9999L);
        NeuralNetwork netB = builderB.build();

        // Fresh network B has different predictions prior to loading
        boolean initialPredictionsDiffer = false;
        for (int i = 0; i < testSamples.size(); i++) {
            double[] outB = netB.getOutput(testSamples.get(i));
            double[] outA = outputsA.get(i);
            for (int k = 0; k < outA.length; k++) {
                if (Math.abs(outA[k] - outB[k]) > 1e-4) {
                    initialPredictionsDiffer = true;
                    break;
                }
            }
        }
        assertTrue("Fresh Network B with seed 9999 initially differs from trained Network A", initialPredictionsDiffer);

        // 4. Reload weights into Network B
        ModelSerializer.loadWeights(netB, tempFile);

        // 5. Verify Network B outputs are now 100% IDENTICAL to Network A
        float accuracyB = netB.test(testSamples);
        assertTrue("Accuracy of reloaded Network B matches Network A exactly (" + (accuracyA * 100) + "%)",
                Float.compare(accuracyA, accuracyB) == 0);

        boolean allOutputsIdentical = true;
        for (int i = 0; i < testSamples.size(); i++) {
            double[] outB = netB.getOutput(testSamples.get(i));
            double[] outA = outputsA.get(i);
            int guessB = netB.guess(testSamples.get(i));
            int guessA = guessesA.get(i);

            if (guessA != guessB) {
                allOutputsIdentical = false;
            }
            for (int k = 0; k < outA.length; k++) {
                if (Double.compare(outA[k], outB[k]) != 0) {
                    allOutputsIdentical = false;
                }
            }
        }
        assertTrue("All 30 test sample predictions and vectors 100% bit-identical between trained and reloaded models",
                allOutputsIdentical);
    }

    private static void testModelSerializerLoadFromScratch() throws IOException {
        System.out.println("\n-- Testing TASK-201: ModelSerializer.load(File) from Scratch --");

        File tempFile = File.createTempFile("cnn_scratch_model_", ".bin");
        tempFile.deleteOnExit();

        NetworkBuilder builder = new NetworkBuilder(AppConfig.INPUT_ROWS, AppConfig.INPUT_COLS, 25600.0);
        builder.addConvolutionLayer(6, 5, 1, 0.05, 555L);
        builder.addMaxPoolLayer(3, 2);
        builder.addFullyConnectedLayer(10, 0.05, 555L);
        NeuralNetwork original = builder.build();

        ModelSerializer.save(original, tempFile);

        NeuralNetwork reconstructed = ModelSerializer.load(tempFile);
        assertTrue("Reconstructed network non-null", reconstructed != null);
        assertTrue("Reconstructed network has 3 layers", reconstructed.getLayers().size() == 3);

        ConvolutionLayer conv = (ConvolutionLayer) reconstructed.getLayers().get(0);
        assertTrue("Reconstructed conv has 6 filters", conv.getNumFilters() == 6);
        assertTrue("Reconstructed conv filter size is 5", conv.getFilterSize() == 5);

        FullyConnectedLayer fc = (FullyConnectedLayer) reconstructed.getLayers().get(2);
        assertTrue("Reconstructed fc out length is 10", fc.getOutLength() == 10);
    }

    private static void testDimensionalMismatchValidation() throws IOException {
        System.out.println("\n-- Testing TASK-201: Dimensional Mismatch Diagnostic Errors (Rule 2.2) --");

        // Save a model with 4 filters
        File modelFile = File.createTempFile("cnn_dim_mismatch_", ".bin");
        modelFile.deleteOnExit();

        NetworkBuilder builder4 = new NetworkBuilder(AppConfig.INPUT_ROWS, AppConfig.INPUT_COLS, 25600.0);
        builder4.addConvolutionLayer(4, 5, 1, 0.1, 1L);
        builder4.addMaxPoolLayer(3, 2);
        builder4.addFullyConnectedLayer(10, 0.1, 1L);
        NeuralNetwork net4 = builder4.build();
        ModelSerializer.save(net4, modelFile);

        // Attempt to load into a network expecting 8 filters
        NetworkBuilder builder8 = new NetworkBuilder(AppConfig.INPUT_ROWS, AppConfig.INPUT_COLS, 25600.0);
        builder8.addConvolutionLayer(8, 5, 1, 0.1, 1L);
        builder8.addMaxPoolLayer(3, 2);
        builder8.addFullyConnectedLayer(10, 0.1, 1L);
        NeuralNetwork net8 = builder8.build();

        try {
            ModelSerializer.loadWeights(net8, modelFile);
            assertTrue("Loading mismatched filter count should fail", false);
        } catch (IOException e) {
            assertTrue("Filter mismatch threw descriptive IOException: " + e.getMessage(),
                    e.getMessage().contains("ConvolutionLayer dimension mismatch"));
        }
    }

    private static void testCorruptedFileAndInvalidHeaderHandling() throws IOException {
        System.out.println("\n-- Testing TASK-201: Corrupted File & Header Validation --");

        // 1. Invalid Magic
        File badMagicFile = File.createTempFile("bad_magic_", ".bin");
        badMagicFile.deleteOnExit();
        try (DataOutputStream dos = new DataOutputStream(new FileOutputStream(badMagicFile))) {
            dos.writeInt(0xDEADBEEF); // Bad magic
            dos.writeShort((short) 1);
            dos.writeDouble(25600.0);
            dos.writeInt(3);
        }

        try {
            ModelSerializer.load(badMagicFile);
            assertTrue("Bad magic should throw IOException", false);
        } catch (IOException e) {
            assertTrue("Bad magic caught: " + e.getMessage(), e.getMessage().contains("Invalid model file magic"));
        }

        // 2. Unsupported Version
        File badVersionFile = File.createTempFile("bad_version_", ".bin");
        badVersionFile.deleteOnExit();
        try (DataOutputStream dos = new DataOutputStream(new FileOutputStream(badVersionFile))) {
            dos.writeInt(ModelSerializer.MAGIC);
            dos.writeShort((short) 99); // Unsupported version
            dos.writeDouble(25600.0);
            dos.writeInt(3);
        }

        try {
            ModelSerializer.load(badVersionFile);
            assertTrue("Unsupported version should throw IOException", false);
        } catch (IOException e) {
            assertTrue("Unsupported version caught: " + e.getMessage(), e.getMessage().contains("Unsupported model format version"));
        }

        // 3. Nonexistent file
        try {
            ModelSerializer.load(new File("nonexistent_path_file_12345.bin"));
            assertTrue("Nonexistent file should throw IOException", false);
        } catch (IOException e) {
            assertTrue("Nonexistent file threw IOException: " + e.getMessage(), true);
        }
    }

    private static void testEventDispatchThreadProtection() throws Exception {
        System.out.println("\n-- Testing Rule 1.3: Non-Blocking GUI Threading & EDT Protection --");

        File tempFile = File.createTempFile("edt_test_", ".bin");
        tempFile.deleteOnExit();

        NetworkBuilder builder = new NetworkBuilder(AppConfig.INPUT_ROWS, AppConfig.INPUT_COLS, 25600.0);
        builder.addConvolutionLayer(2, 5, 1, 0.1, 1L);
        builder.addMaxPoolLayer(3, 2);
        builder.addFullyConnectedLayer(10, 0.1, 1L);
        NeuralNetwork net = builder.build();

        AtomicBoolean saveBlockedOnEdt = new AtomicBoolean(false);
        AtomicBoolean loadBlockedOnEdt = new AtomicBoolean(false);

        // Attempt save on Swing EDT
        SwingUtilities.invokeAndWait(() -> {
            try {
                ModelSerializer.save(net, tempFile);
            } catch (IllegalStateException e) {
                if (e.getMessage().contains("cannot be executed on the Swing Event Dispatch Thread (EDT)")) {
                    saveBlockedOnEdt.set(true);
                }
            } catch (Exception ignored) {}

            try {
                ModelSerializer.load(tempFile);
            } catch (IllegalStateException e) {
                if (e.getMessage().contains("cannot be executed on the Swing Event Dispatch Thread (EDT)")) {
                    loadBlockedOnEdt.set(true);
                }
            } catch (Exception ignored) {}
        });

        assertTrue("ModelSerializer.save threw IllegalStateException on EDT", saveBlockedOnEdt.get());
        assertTrue("ModelSerializer.load threw IllegalStateException on EDT", loadBlockedOnEdt.get());
    }

    private static void testAsyncModelPersistenceService() throws Exception {
        System.out.println("\n-- Testing service.ModelPersistenceService Asynchronous Execution --");

        File tempFile = File.createTempFile("async_model_", ".bin");
        tempFile.deleteOnExit();

        NetworkBuilder builder = new NetworkBuilder(AppConfig.INPUT_ROWS, AppConfig.INPUT_COLS, 25600.0);
        builder.addConvolutionLayer(2, 5, 1, 0.1, 77L);
        builder.addMaxPoolLayer(3, 2);
        builder.addFullyConnectedLayer(10, 0.1, 77L);
        NeuralNetwork net = builder.build();

        // 1. Test saveAsync with CompletableFuture
        CompletableFuture<Void> saveFuture = ModelPersistenceService.saveAsync(net, tempFile);
        saveFuture.get(5, TimeUnit.SECONDS);
        assertTrue("saveAsync CompletableFuture completed within timeout", tempFile.exists() && tempFile.length() > 0);

        // 2. Test loadAsync with CompletableFuture
        CompletableFuture<NeuralNetwork> loadFuture = ModelPersistenceService.loadAsync(tempFile);
        NeuralNetwork loadedNet = loadFuture.get(5, TimeUnit.SECONDS);
        assertTrue("loadAsync CompletableFuture loaded network", loadedNet != null);
        assertTrue("Loaded network layer count is 3", loadedNet.getLayers().size() == 3);

        // 3. Test PersistenceCallback with loadWeightsAsync
        CountDownLatch callbackLatch = new CountDownLatch(1);
        AtomicBoolean callbackSuccess = new AtomicBoolean(false);

        NetworkBuilder builderTarget = new NetworkBuilder(AppConfig.INPUT_ROWS, AppConfig.INPUT_COLS, 25600.0);
        builderTarget.addConvolutionLayer(2, 5, 1, 0.1, 999L);
        builderTarget.addMaxPoolLayer(3, 2);
        builderTarget.addFullyConnectedLayer(10, 0.1, 999L);
        NeuralNetwork targetNet = builderTarget.build();

        ModelPersistenceService.loadWeightsAsync(targetNet, tempFile, new ModelPersistenceService.PersistenceCallback<Void>() {
            @Override
            public void onSuccess(Void result) {
                callbackSuccess.set(true);
                callbackLatch.countDown();
            }

            @Override
            public void onFailure(Throwable error) {
                callbackLatch.countDown();
            }
        });

        boolean completed = callbackLatch.await(5, TimeUnit.SECONDS);
        assertTrue("PersistenceCallback invoked within timeout", completed);
        assertTrue("PersistenceCallback reported success", callbackSuccess.get());
    }

    private static void testMainExecuteIntegration() throws Exception {
        System.out.println("\n-- Testing Main.execute CLI Pipeline with --save and --load --");

        File savedModel = File.createTempFile("main_model_test_", ".bin");
        savedModel.deleteOnExit();

        // Step 1: Run Main with quick mode, 1 epoch, and --save
        String[] saveArgs = new String[]{
                "--quick",
                "--epochs", "1",
                "--filters", "4",
                "--save", savedModel.getAbsolutePath()
        };
        Class<?> mainClass = Class.forName("Main");
        java.lang.reflect.Method executeMethod = mainClass.getMethod("execute", String[].class);
        executeMethod.invoke(null, (Object) saveArgs);

        assertTrue("Main.execute generated save file", savedModel.exists() && savedModel.length() > 0);

        // Step 2: Run Main with quick mode, --epochs 0 (eval only), and --load
        String[] loadArgs = new String[]{
                "--quick",
                "--epochs", "0",
                "--filters", "4",
                "--load", savedModel.getAbsolutePath()
        };
        executeMethod.invoke(null, (Object) loadArgs);
        assertTrue("Main.execute ran evaluation mode with --load cleanly", true);
    }
}
