package network;

import layers.ConvolutionLayer;
import layers.FullyConnectedLayer;
import layers.Layer;
import layers.MaxPoolLayer;

import javax.swing.SwingUtilities;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * High-performance, pure Core Java binary serializer and deserializer for
 * {@link NeuralNetwork} models and layer weights.
 *
 * <p>Persists trained convolution filters and dense layer weight matrices along
 * with architectural metadata, ensuring exact reproducibility and rapid model reloads
 * without retraining.</p>
 *
 * <p>Guards against blocking disk I/O on the Swing Event Dispatch Thread (EDT)
 * as mandated by architectural rules.</p>
 */
public final class ModelSerializer {

    /** Magic identifier: "CNNW" in ASCII (0x434E4E57) */
    public static final int MAGIC = 0x434E4E57;

    /** Binary format specification version */
    public static final short VERSION = 1;

    /** Layer type tag constants */
    public static final int TAG_CONV = 1;
    public static final int TAG_MAX_POOL = 2;
    public static final int TAG_FULLY_CONNECTED = 3;

    private ModelSerializer() {}

    /**
     * Serializes the neural network architecture and trained weights to the specified file.
     *
     * @param net Model to serialize
     * @param file Target output file
     * @throws IOException If writing fails
     * @throws IllegalStateException If invoked on the Swing Event Dispatch Thread
     */
    public static void save(NeuralNetwork net, File file) throws IOException {
        Objects.requireNonNull(net, "NeuralNetwork cannot be null");
        Objects.requireNonNull(file, "Target file cannot be null");
        checkNotEventDispatchThread("save");

        File parent = file.getParentFile();
        if (parent != null && !parent.exists()) {
            parent.mkdirs();
        }

        try (OutputStream fos = new FileOutputStream(file);
             BufferedOutputStream bos = new BufferedOutputStream(fos)) {
            save(net, bos);
        } catch (IOException e) {
            throw new IOException("Failed to save model weights to: " + file.getAbsolutePath(), e);
        }
    }

    /**
     * Serializes the neural network architecture and trained weights to the specified path.
     *
     * @param net Model to serialize
     * @param filePath Target output file path
     * @throws IOException If writing fails
     */
    public static void save(NeuralNetwork net, String filePath) throws IOException {
        Objects.requireNonNull(filePath, "File path cannot be null");
        save(net, new File(filePath));
    }

    /**
     * Serializes the neural network architecture and trained weights to an OutputStream.
     *
     * @param net Model to serialize
     * @param out Target output stream
     * @throws IOException If writing fails
     */
    public static void save(NeuralNetwork net, OutputStream out) throws IOException {
        Objects.requireNonNull(net, "NeuralNetwork cannot be null");
        Objects.requireNonNull(out, "OutputStream cannot be null");
        checkNotEventDispatchThread("save");

        DataOutputStream dos = (out instanceof DataOutputStream)
                ? (DataOutputStream) out
                : new DataOutputStream(out);

        List<Layer> layers = net.getLayers();
        if (layers == null || layers.isEmpty()) {
            throw new IllegalArgumentException("NeuralNetwork contains no layers to serialize.");
        }

        // Header
        dos.writeInt(MAGIC);
        dos.writeShort(VERSION);
        dos.writeDouble(net.getScaleFactor());
        dos.writeInt(layers.size());

        // Layers
        for (int i = 0; i < layers.size(); i++) {
            Layer layer = layers.get(i);
            if (layer instanceof ConvolutionLayer) {
                ConvolutionLayer conv = (ConvolutionLayer) layer;
                dos.writeInt(TAG_CONV);
                dos.writeInt(conv.getFilterSize());
                dos.writeInt(conv.getStepSize());
                dos.writeInt(conv.getInLength());
                dos.writeInt(conv.getInRows());
                dos.writeInt(conv.getInCols());
                dos.writeDouble(conv.getLearningRate());

                List<double[][]> filters = conv.getFilters();
                int numFilters = (filters != null) ? filters.size() : 0;
                dos.writeInt(numFilters);

                int filterSize = conv.getFilterSize();
                for (int f = 0; f < numFilters; f++) {
                    double[][] filter = filters.get(f);
                    for (int r = 0; r < filterSize; r++) {
                        for (int c = 0; c < filterSize; c++) {
                            dos.writeDouble(filter[r][c]);
                        }
                    }
                }
            } else if (layer instanceof MaxPoolLayer) {
                MaxPoolLayer pool = (MaxPoolLayer) layer;
                dos.writeInt(TAG_MAX_POOL);
                dos.writeInt(pool.getStepSize());
                dos.writeInt(pool.getWindowSize());
                dos.writeInt(pool.getInLength());
                dos.writeInt(pool.getInRows());
                dos.writeInt(pool.getInCols());
            } else if (layer instanceof FullyConnectedLayer) {
                FullyConnectedLayer fc = (FullyConnectedLayer) layer;
                dos.writeInt(TAG_FULLY_CONNECTED);
                int inLength = fc.getInLength();
                int outLength = fc.getOutLength();
                dos.writeInt(inLength);
                dos.writeInt(outLength);
                dos.writeDouble(fc.getLearningRate());

                double[][] weights = fc.getWeights();
                for (int r = 0; r < inLength; r++) {
                    for (int c = 0; c < outLength; c++) {
                        dos.writeDouble(weights[r][c]);
                    }
                }
            } else {
                throw new IOException("Unsupported layer type encountered at index " + i + ": "
                        + layer.getClass().getName());
            }
        }

        dos.flush();
    }

    /**
     * Deserializes and constructs a complete {@link NeuralNetwork} from a model file.
     *
     * @param file Serialized model file
     * @return Freshly reconstructed NeuralNetwork ready for inference or training
     * @throws IOException If parsing or reading fails
     */
    public static NeuralNetwork load(File file) throws IOException {
        Objects.requireNonNull(file, "Source file cannot be null");
        checkNotEventDispatchThread("load");

        if (!file.exists()) {
            throw new FileNotFoundException("Model file does not exist: " + file.getAbsolutePath());
        }

        try (InputStream fis = new FileInputStream(file);
             BufferedInputStream bis = new BufferedInputStream(fis)) {
            return load(bis);
        } catch (IOException e) {
            throw new IOException("Failed to load model from: " + file.getAbsolutePath() + " (" + e.getMessage() + ")", e);
        }
    }

    /**
     * Deserializes and constructs a complete {@link NeuralNetwork} from a model file path.
     *
     * @param filePath Serialized model file path
     * @return Freshly reconstructed NeuralNetwork
     * @throws IOException If parsing or reading fails
     */
    public static NeuralNetwork load(String filePath) throws IOException {
        Objects.requireNonNull(filePath, "File path cannot be null");
        return load(new File(filePath));
    }

    /**
     * Deserializes and constructs a complete {@link NeuralNetwork} from an InputStream.
     *
     * @param in Input stream containing serialized model bytes
     * @return Freshly reconstructed NeuralNetwork
     * @throws IOException If parsing or reading fails
     */
    public static NeuralNetwork load(InputStream in) throws IOException {
        Objects.requireNonNull(in, "InputStream cannot be null");
        checkNotEventDispatchThread("load");

        DataInputStream dis = (in instanceof DataInputStream)
                ? (DataInputStream) in
                : new DataInputStream(in);

        int magic = dis.readInt();
        if (magic != MAGIC) {
            throw new IOException(String.format(
                    "Invalid model file magic: expected 0x%08X ('CNNW'), got 0x%08X", MAGIC, magic));
        }

        short version = dis.readShort();
        if (version != VERSION) {
            throw new IOException(String.format(
                    "Unsupported model format version: expected %d, got %d", VERSION, version));
        }

        double scaleFactor = dis.readDouble();
        int numLayers = dis.readInt();
        if (numLayers <= 0) {
            throw new IOException("Invalid layer count in model file: " + numLayers);
        }

        List<Layer> layers = new ArrayList<>(numLayers);

        for (int l = 0; l < numLayers; l++) {
            int tag = dis.readInt();
            if (tag == TAG_CONV) {
                int filterSize = dis.readInt();
                int stepSize = dis.readInt();
                int inLength = dis.readInt();
                int inRows = dis.readInt();
                int inCols = dis.readInt();
                double learningRate = dis.readDouble();
                int numFilters = dis.readInt();

                List<double[][]> filters = new ArrayList<>(numFilters);
                for (int f = 0; f < numFilters; f++) {
                    double[][] filter = new double[filterSize][filterSize];
                    for (int r = 0; r < filterSize; r++) {
                        for (int c = 0; c < filterSize; c++) {
                            filter[r][c] = dis.readDouble();
                        }
                    }
                    filters.add(filter);
                }

                ConvolutionLayer conv = new ConvolutionLayer(
                        filterSize, stepSize, inLength, inRows, inCols, learningRate, filters);
                layers.add(conv);
            } else if (tag == TAG_MAX_POOL) {
                int stepSize = dis.readInt();
                int windowSize = dis.readInt();
                int inLength = dis.readInt();
                int inRows = dis.readInt();
                int inCols = dis.readInt();

                MaxPoolLayer pool = new MaxPoolLayer(stepSize, windowSize, inLength, inRows, inCols);
                layers.add(pool);
            } else if (tag == TAG_FULLY_CONNECTED) {
                int inLength = dis.readInt();
                int outLength = dis.readInt();
                double learningRate = dis.readDouble();

                double[][] weights = new double[inLength][outLength];
                for (int r = 0; r < inLength; r++) {
                    for (int c = 0; c < outLength; c++) {
                        weights[r][c] = dis.readDouble();
                    }
                }

                FullyConnectedLayer fc = new FullyConnectedLayer(inLength, outLength, learningRate, weights);
                layers.add(fc);
            } else {
                throw new IOException(String.format(
                        "Unknown layer type tag %d encountered at index %d", tag, l));
            }
        }

        return new NeuralNetwork(layers, scaleFactor);
    }

    /**
     * Loads weights from a serialized model file into an existing {@link NeuralNetwork}.
     * Validates that layer types and dimensions strictly match the destination network.
     *
     * @param net Target network to populate with deserialized weights
     * @param file Serialized model file
     * @throws IOException If file reading or dimension validation fails
     */
    public static void loadWeights(NeuralNetwork net, File file) throws IOException {
        Objects.requireNonNull(net, "NeuralNetwork cannot be null");
        Objects.requireNonNull(file, "Source file cannot be null");
        checkNotEventDispatchThread("loadWeights");

        if (!file.exists()) {
            throw new FileNotFoundException("Model file does not exist: " + file.getAbsolutePath());
        }

        try (InputStream fis = new FileInputStream(file);
             BufferedInputStream bis = new BufferedInputStream(fis)) {
            loadWeights(net, bis);
        } catch (IOException e) {
            throw new IOException("Failed to load weights into network from: " + file.getAbsolutePath() + " (" + e.getMessage() + ")", e);
        }
    }

    /**
     * Loads weights from a serialized model file path into an existing {@link NeuralNetwork}.
     *
     * @param net Target network to populate with deserialized weights
     * @param filePath Serialized model file path
     * @throws IOException If file reading or dimension validation fails
     */
    public static void loadWeights(NeuralNetwork net, String filePath) throws IOException {
        Objects.requireNonNull(filePath, "File path cannot be null");
        loadWeights(net, new File(filePath));
    }

    /**
     * Loads weights from an InputStream into an existing {@link NeuralNetwork}.
     * Validates layer types and dimensions against the target network.
     *
     * @param net Target network to populate with deserialized weights
     * @param in Input stream containing serialized model bytes
     * @throws IOException If reading fails or dimensions do not match
     */
    public static void loadWeights(NeuralNetwork net, InputStream in) throws IOException {
        Objects.requireNonNull(net, "NeuralNetwork cannot be null");
        Objects.requireNonNull(in, "InputStream cannot be null");
        checkNotEventDispatchThread("loadWeights");

        DataInputStream dis = (in instanceof DataInputStream)
                ? (DataInputStream) in
                : new DataInputStream(in);

        int magic = dis.readInt();
        if (magic != MAGIC) {
            throw new IOException(String.format(
                    "Invalid model file magic: expected 0x%08X ('CNNW'), got 0x%08X", MAGIC, magic));
        }

        short version = dis.readShort();
        if (version != VERSION) {
            throw new IOException(String.format(
                    "Unsupported model format version: expected %d, got %d", VERSION, version));
        }

        double scaleFactor = dis.readDouble();
        int numLayers = dis.readInt();

        List<Layer> targetLayers = net.getLayers();
        if (targetLayers == null || targetLayers.size() != numLayers) {
            int actualCount = (targetLayers == null) ? 0 : targetLayers.size();
            throw new IOException(String.format(
                    "Layer count mismatch: model file has %d layers, but target network has %d layers",
                    numLayers, actualCount));
        }

        net.setScaleFactor(scaleFactor);

        for (int l = 0; l < numLayers; l++) {
            Layer targetLayer = targetLayers.get(l);
            int tag = dis.readInt();

            if (tag == TAG_CONV) {
                if (!(targetLayer instanceof ConvolutionLayer)) {
                    throw new IOException(String.format(
                            "Layer %d type mismatch: model file has ConvolutionLayer, but network has %s",
                            l, targetLayer.getClass().getSimpleName()));
                }
                ConvolutionLayer conv = (ConvolutionLayer) targetLayer;

                int filterSize = dis.readInt();
                int stepSize = dis.readInt();
                int inLength = dis.readInt();
                int inRows = dis.readInt();
                int inCols = dis.readInt();
                double learningRate = dis.readDouble();
                int numFilters = dis.readInt();

                if (conv.getFilterSize() != filterSize || conv.getStepSize() != stepSize ||
                        conv.getNumFilters() != numFilters || conv.getInRows() != inRows ||
                        conv.getInCols() != inCols || conv.getInLength() != inLength) {
                    throw new IOException(String.format(
                            "ConvolutionLayer dimension mismatch at layer %d: file has [filters: %d, size: %dx%d, step: %d, in: %dx%dx%d], " +
                            "network has [filters: %d, size: %dx%d, step: %d, in: %dx%dx%d]",
                            l, numFilters, filterSize, filterSize, stepSize, inLength, inRows, inCols,
                            conv.getNumFilters(), conv.getFilterSize(), conv.getFilterSize(), conv.getStepSize(),
                            conv.getInLength(), conv.getInRows(), conv.getInCols()));
                }

                List<double[][]> filters = new ArrayList<>(numFilters);
                for (int f = 0; f < numFilters; f++) {
                    double[][] filter = new double[filterSize][filterSize];
                    for (int r = 0; r < filterSize; r++) {
                        for (int c = 0; c < filterSize; c++) {
                            filter[r][c] = dis.readDouble();
                        }
                    }
                    filters.add(filter);
                }
                conv.setFilters(filters);

            } else if (tag == TAG_MAX_POOL) {
                if (!(targetLayer instanceof MaxPoolLayer)) {
                    throw new IOException(String.format(
                            "Layer %d type mismatch: model file has MaxPoolLayer, but network has %s",
                            l, targetLayer.getClass().getSimpleName()));
                }
                MaxPoolLayer pool = (MaxPoolLayer) targetLayer;

                int stepSize = dis.readInt();
                int windowSize = dis.readInt();
                int inLength = dis.readInt();
                int inRows = dis.readInt();
                int inCols = dis.readInt();

                if (pool.getWindowSize() != windowSize || pool.getStepSize() != stepSize ||
                        pool.getInLength() != inLength || pool.getInRows() != inRows || pool.getInCols() != inCols) {
                    throw new IOException(String.format(
                            "MaxPoolLayer dimension mismatch at layer %d: file has [window: %dx%d, step: %d, in: %dx%dx%d], " +
                            "network has [window: %dx%d, step: %d, in: %dx%dx%d]",
                            l, windowSize, windowSize, stepSize, inLength, inRows, inCols,
                            pool.getWindowSize(), pool.getWindowSize(), pool.getStepSize(),
                            pool.getInLength(), pool.getInRows(), pool.getInCols()));
                }

            } else if (tag == TAG_FULLY_CONNECTED) {
                if (!(targetLayer instanceof FullyConnectedLayer)) {
                    throw new IOException(String.format(
                            "Layer %d type mismatch: model file has FullyConnectedLayer, but network has %s",
                            l, targetLayer.getClass().getSimpleName()));
                }
                FullyConnectedLayer fc = (FullyConnectedLayer) targetLayer;

                int inLength = dis.readInt();
                int outLength = dis.readInt();
                double learningRate = dis.readDouble();

                if (fc.getInLength() != inLength || fc.getOutLength() != outLength) {
                    throw new IOException(String.format(
                            "FullyConnectedLayer dimension mismatch at layer %d: file has [%d -> %d], " +
                            "network has [%d -> %d]",
                            l, inLength, outLength, fc.getInLength(), fc.getOutLength()));
                }

                double[][] weights = new double[inLength][outLength];
                for (int r = 0; r < inLength; r++) {
                    for (int c = 0; c < outLength; c++) {
                        weights[r][c] = dis.readDouble();
                    }
                }
                fc.setWeights(weights);

            } else {
                throw new IOException(String.format(
                        "Unknown layer type tag %d encountered at index %d", tag, l));
            }
        }
    }

    private static void checkNotEventDispatchThread(String operation) {
        if (SwingUtilities.isEventDispatchThread()) {
            throw new IllegalStateException(String.format(
                    "Disk I/O operation '%s' cannot be executed on the Swing Event Dispatch Thread (EDT). " +
                    "Use ModelPersistenceService for asynchronous persistence.", operation));
        }
    }
}
