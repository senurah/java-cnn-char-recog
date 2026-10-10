package cli;

import java.io.PrintStream;

/**
 * Educational parameter dictionary explaining the architectural role,
 * computational trade-offs, and recommended ranges for CNN hyperparameters.
 */
public final class ParameterExplainer {

    public static class ParameterDoc {
        private final String name;
        private final String meaning;
        private final String impact;
        private final String recommendedRange;

        public ParameterDoc(String name, String meaning, String impact, String recommendedRange) {
            this.name = name;
            this.meaning = meaning;
            this.impact = impact;
            this.recommendedRange = recommendedRange;
        }

        public String getName() { return name; }
        public String getMeaning() { return meaning; }
        public String getImpact() { return impact; }
        public String getRecommendedRange() { return recommendedRange; }

        public String toFormattedString() {
            return String.format(
                    "------------------------------------------------------------------------\n" +
                    "[Parameter: %s]\n" +
                    "  Meaning          : %s\n" +
                    "  Impact           : %s\n" +
                    "  Recommended Range: %s\n" +
                    "------------------------------------------------------------------------",
                    name, meaning, impact, recommendedRange);
        }
    }

    public static final ParameterDoc EPOCHS = new ParameterDoc(
            "Training Epochs",
            "The total number of complete passes through the training dataset.",
            "More epochs allow deeper feature optimization but linearly scale training time and may risk overfitting.",
            "1 to 20 (Max: 1000)");

    public static final ParameterDoc NUM_FILTERS = new ParameterDoc(
            "Convolution Filters",
            "The number of distinct 2D feature detectors (e.g. edge, corner, curve detectors).",
            "More filters increase representational power and pattern variety, but scale computation and memory linearly.",
            "4 to 32 (Default: 8)");

    public static final ParameterDoc FILTER_SIZE = new ParameterDoc(
            "Kernel / Filter Size",
            "The spatial dimension (N x N) of the sliding convolution receptive field.",
            "Larger kernels capture broader spatial structures; smaller kernels capture fine local textures. Cannot exceed 28x28.",
            "3 to 7 (Default: 5)");

    public static final ParameterDoc CONV_STRIDE = new ParameterDoc(
            "Convolution Stride",
            "The step size in pixels by which the kernel shifts horizontally and vertically across the image.",
            "A larger stride downsamples output feature maps faster, decreasing spatial resolution and computation.",
            "1 to 2 (Default: 1)");

    public static final ParameterDoc CONV_LR = new ParameterDoc(
            "Convolution Learning Rate",
            "The gradient descent step multiplier used when updating convolution filter weights.",
            "Too large causes gradient divergence; too small leads to slow convergence. Must be in range (0.0, 1.0].",
            "0.01 to 0.2 (Default: 0.1)");

    public static final ParameterDoc POOL_WINDOW = new ParameterDoc(
            "Max Pooling Window Size",
            "The spatial dimension (N x N) over which the maximum activation is pooled.",
            "Downsamples feature map dimensions and provides local spatial translation invariance. Must not exceed conv map size.",
            "2 to 4 (Default: 3)");

    public static final ParameterDoc POOL_STRIDE = new ParameterDoc(
            "Max Pooling Stride",
            "The step size by which the pooling window slides across feature maps.",
            "Controls the spatial reduction factor. Must ensure output dimensions remain positive.",
            "1 to 2 (Default: 2)");

    public static final ParameterDoc NUM_CLASSES = new ParameterDoc(
            "Output Classes",
            "The number of target classification categories in the final output layer.",
            "Determines dense layer output dimension (digits 0-9 for MNIST = 10 classes).",
            "10 (Fixed for MNIST digit dataset)");

    public static final ParameterDoc FC_LR = new ParameterDoc(
            "Fully Connected Learning Rate",
            "The gradient descent step multiplier used when updating dense classification weights.",
            "Controls classification weight adjustment magnitude during backpropagation. Must be in range (0.0, 1.0].",
            "0.01 to 0.2 (Default: 0.1)");

    public static final ParameterDoc SCALE_FACTOR = new ParameterDoc(
            "Input Normalization Scale Factor",
            "The divisor applied to raw grayscale pixel values [0..255] before feeding to the network.",
            "Normalizes input activations to prevent gradient explosion and neuron saturation.",
            "256.0 to 25600.0 (Default: 25600.0)");

    public static final ParameterDoc SEED = new ParameterDoc(
            "Random Seed",
            "The pseudorandom seed for Gaussian filter initialization and reproducible weight setup.",
            "Guarantees deterministic, reproducible initial weights across distinct runs.",
            "Any integer or long (Default: 123)");

    public static final ParameterDoc TRAIN_LIMIT = new ParameterDoc(
            "Training Sample Limit",
            "Maximum number of training samples to load from the dataset (0 = all 60,000).",
            "Use a smaller slice (e.g. 200) for fast verification or 0 for complete full-scale training.",
            "0 (all) or 50 to 60000");

    public static final ParameterDoc TEST_LIMIT = new ParameterDoc(
            "Testing Sample Limit",
            "Maximum number of testing samples to load from the dataset (0 = all 10,000).",
            "Use a smaller slice (e.g. 100) for rapid accuracy checks or 0 for full testing set evaluation.",
            "0 (all) or 50 to 10000");

    public static final ParameterDoc SAVE_PATH = new ParameterDoc(
            "Model Save Path",
            "File path where trained convolution filters and dense layer weights are serialized.",
            "Enables reusing trained models for inference or fine-tuning without retraining from scratch.",
            "Valid filesystem path (e.g. models/mnist_cnn.bin)");

    public static final ParameterDoc LOAD_PATH = new ParameterDoc(
            "Model Load Path",
            "File path from which pre-trained CNN weights are loaded into the network.",
            "Bypasses training from random scratch weights, allowing immediate high-accuracy evaluation or transfer learning.",
            "Valid existing model file path");

    private ParameterExplainer() {}

    public static void printDoc(PrintStream out, ParameterDoc doc) {
        out.println(doc.toFormattedString());
    }

    public static void printAllDocs(PrintStream out) {
        out.println("========================================================================");
        out.println("                     CNN HYPERPARAMETER EDUCATIONAL GUIDE                ");
        out.println("========================================================================");
        printDoc(out, EPOCHS);
        printDoc(out, NUM_FILTERS);
        printDoc(out, FILTER_SIZE);
        printDoc(out, CONV_STRIDE);
        printDoc(out, CONV_LR);
        printDoc(out, POOL_WINDOW);
        printDoc(out, POOL_STRIDE);
        printDoc(out, NUM_CLASSES);
        printDoc(out, FC_LR);
        printDoc(out, SCALE_FACTOR);
        printDoc(out, SEED);
        printDoc(out, TRAIN_LIMIT);
        printDoc(out, TEST_LIMIT);
        printDoc(out, SAVE_PATH);
        printDoc(out, LOAD_PATH);
        out.println("========================================================================");
    }
}
