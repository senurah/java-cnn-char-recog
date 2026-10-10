package service;

import network.ModelSerializer;
import network.NeuralNetwork;

import javax.swing.SwingUtilities;
import java.awt.GraphicsEnvironment;
import java.io.File;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Service managing asynchronous, non-blocking model persistence.
 *
 * <p>Ensures disk I/O operations are offloaded from the calling thread—strictly
 * preventing any execution on the Swing Event Dispatch Thread (EDT) as required
 * by architectural standards.</p>
 *
 * <p>Supports {@link CompletableFuture}-based composition as well as GUI-friendly
 * {@link PersistenceCallback} event listeners dispatched onto the EDT.</p>
 */
public final class ModelPersistenceService {

    /**
     * Callback interface for asynchronous persistence operations.
     * Notifications are automatically routed to the Swing Event Dispatch Thread (EDT)
     * when a GUI desktop environment is active.
     *
     * @param <T> Result payload type
     */
    public interface PersistenceCallback<T> {
        /**
         * Invoked upon successful completion of the persistence operation.
         *
         * @param result Result value (or null for void operations)
         */
        void onSuccess(T result);

        /**
         * Invoked when persistence encounters an I/O or validation failure.
         *
         * @param error Root cause exception
         */
        void onFailure(Throwable error);
    }

    private static final ThreadFactory THREAD_FACTORY = new ThreadFactory() {
        private final AtomicInteger counter = new AtomicInteger(1);

        @Override
        public Thread newThread(Runnable r) {
            Thread t = new Thread(r, "model-persistence-worker-" + counter.getAndIncrement());
            t.setDaemon(true);
            return t;
        }
    };

    private static final ExecutorService EXECUTOR = Executors.newCachedThreadPool(THREAD_FACTORY);

    private ModelPersistenceService() {}

    /**
     * Asynchronously serializes a neural network to the specified file.
     *
     * @param net Model to persist
     * @param file Target file
     * @return CompletableFuture completing when persistence is finished
     */
    public static CompletableFuture<Void> saveAsync(NeuralNetwork net, File file) {
        return saveAsync(net, file, null);
    }

    /**
     * Asynchronously serializes a neural network to the specified file path.
     *
     * @param net Model to persist
     * @param filePath Target file path
     * @return CompletableFuture completing when persistence is finished
     */
    public static CompletableFuture<Void> saveAsync(NeuralNetwork net, String filePath) {
        Objects.requireNonNull(filePath, "File path cannot be null");
        return saveAsync(net, new File(filePath), null);
    }

    /**
     * Asynchronously serializes a neural network to the specified file with a listener callback.
     *
     * @param net Model to persist
     * @param file Target file
     * @param callback Optional listener notified of success or failure
     * @return CompletableFuture completing when persistence is finished
     */
    public static CompletableFuture<Void> saveAsync(NeuralNetwork net, File file, PersistenceCallback<Void> callback) {
        Objects.requireNonNull(net, "NeuralNetwork cannot be null");
        Objects.requireNonNull(file, "Target file cannot be null");

        CompletableFuture<Void> future = CompletableFuture.runAsync(() -> {
            try {
                ModelSerializer.save(net, file);
            } catch (Exception e) {
                throw new PersistenceException("Failed to save model to " + file.getAbsolutePath(), e);
            }
        }, EXECUTOR);

        attachCallback(future, callback);
        return future;
    }

    /**
     * Asynchronously loads and reconstructs a neural network from a model file.
     *
     * @param file Source model file
     * @return CompletableFuture yielding the loaded NeuralNetwork
     */
    public static CompletableFuture<NeuralNetwork> loadAsync(File file) {
        return loadAsync(file, null);
    }

    /**
     * Asynchronously loads and reconstructs a neural network from a model file path.
     *
     * @param filePath Source model file path
     * @return CompletableFuture yielding the loaded NeuralNetwork
     */
    public static CompletableFuture<NeuralNetwork> loadAsync(String filePath) {
        Objects.requireNonNull(filePath, "File path cannot be null");
        return loadAsync(new File(filePath), null);
    }

    /**
     * Asynchronously loads and reconstructs a neural network from a model file with a callback.
     *
     * @param file Source model file
     * @param callback Optional listener notified of success or failure
     * @return CompletableFuture yielding the loaded NeuralNetwork
     */
    public static CompletableFuture<NeuralNetwork> loadAsync(File file, PersistenceCallback<NeuralNetwork> callback) {
        Objects.requireNonNull(file, "Source file cannot be null");

        CompletableFuture<NeuralNetwork> future = CompletableFuture.supplyAsync(() -> {
            try {
                return ModelSerializer.load(file);
            } catch (Exception e) {
                throw new PersistenceException("Failed to load model from " + file.getAbsolutePath(), e);
            }
        }, EXECUTOR);

        attachCallback(future, callback);
        return future;
    }

    /**
     * Asynchronously loads weights from a model file into an existing neural network.
     *
     * @param net Target network to receive weights
     * @param file Source model file
     * @return CompletableFuture completing when weights have been updated
     */
    public static CompletableFuture<Void> loadWeightsAsync(NeuralNetwork net, File file) {
        return loadWeightsAsync(net, file, null);
    }

    /**
     * Asynchronously loads weights from a model file path into an existing neural network.
     *
     * @param net Target network to receive weights
     * @param filePath Source model file path
     * @return CompletableFuture completing when weights have been updated
     */
    public static CompletableFuture<Void> loadWeightsAsync(NeuralNetwork net, String filePath) {
        Objects.requireNonNull(filePath, "File path cannot be null");
        return loadWeightsAsync(net, new File(filePath), null);
    }

    /**
     * Asynchronously loads weights from a model file into an existing neural network with a callback.
     *
     * @param net Target network to receive weights
     * @param file Source model file
     * @param callback Optional listener notified of success or failure
     * @return CompletableFuture completing when weights have been updated
     */
    public static CompletableFuture<Void> loadWeightsAsync(NeuralNetwork net, File file, PersistenceCallback<Void> callback) {
        Objects.requireNonNull(net, "NeuralNetwork cannot be null");
        Objects.requireNonNull(file, "Source file cannot be null");

        CompletableFuture<Void> future = CompletableFuture.runAsync(() -> {
            try {
                ModelSerializer.loadWeights(net, file);
            } catch (Exception e) {
                throw new PersistenceException("Failed to load weights from " + file.getAbsolutePath(), e);
            }
        }, EXECUTOR);

        attachCallback(future, callback);
        return future;
    }

    /**
     * Shuts down the background persistence executor service.
     */
    public static void shutdown() {
        EXECUTOR.shutdown();
    }

    /**
     * Checks whether the background executor service has been shut down.
     */
    public static boolean isShutdown() {
        return EXECUTOR.isShutdown();
    }

    private static <T> void attachCallback(CompletableFuture<T> future, PersistenceCallback<T> callback) {
        if (callback == null) {
            return;
        }

        future.whenComplete((result, throwable) -> {
            Runnable task;
            if (throwable != null) {
                Throwable cause = (throwable.getCause() != null) ? throwable.getCause() : throwable;
                task = () -> callback.onFailure(cause);
            } else {
                task = () -> callback.onSuccess(result);
            }

            dispatchNotification(task);
        });
    }

    private static void dispatchNotification(Runnable task) {
        if (!GraphicsEnvironment.isHeadless()) {
            SwingUtilities.invokeLater(task);
        } else {
            task.run();
        }
    }

    /**
     * Runtime exception wrapping persistence errors during asynchronous execution.
     */
    public static class PersistenceException extends RuntimeException {
        public PersistenceException(String message, Throwable cause) {
            super(message, cause);
        }
    }
}
