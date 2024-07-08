#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#define INPUT_SIZE 784  // 28x28 pixels
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10  // 10 digits
#define LEARNING_RATE 0.01
#define EPOCHS 10
#define BATCH_SIZE 32
#define TRAIN_SIZE 60000
#define TEST_SIZE 10000

// Activation functions
float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

float sigmoid_derivative(float x) {
    return x * (1 - x);
}

// Structure for our neural network
typedef struct {
    float *input_hidden_weights;
    float *hidden_biases;
    float *hidden_output_weights;
    float *output_biases;
} NeuralNetwork;

// Initialize the neural network with random weights
void init_network(NeuralNetwork *nn) {
    nn->input_hidden_weights = (float *)malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    nn->hidden_biases = (float *)malloc(HIDDEN_SIZE * sizeof(float));
    nn->hidden_output_weights = (float *)malloc(HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
    nn->output_biases = (float *)malloc(OUTPUT_SIZE * sizeof(float));

    for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; i++) {
        nn->input_hidden_weights[i] = ((float)rand() / RAND_MAX) * 2 - 1;
    }
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        nn->hidden_biases[i] = ((float)rand() / RAND_MAX) * 2 - 1;
    }
    for (int i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; i++) {
        nn->hidden_output_weights[i] = ((float)rand() / RAND_MAX) * 2 - 1;
    }
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        nn->output_biases[i] = ((float)rand() / RAND_MAX) * 2 - 1;
    }
}

// Forward pass
void forward_pass(NeuralNetwork *nn, float *input, float *hidden, float *output) {
    // Input to hidden layer
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        hidden[i] = 0;
        for (int j = 0; j < INPUT_SIZE; j++) {
            hidden[i] += input[j] * nn->input_hidden_weights[j * HIDDEN_SIZE + i];
        }
        hidden[i] = sigmoid(hidden[i] + nn->hidden_biases[i]);
    }

    // Hidden to output layer
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output[i] = 0;
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            output[i] += hidden[j] * nn->hidden_output_weights[j * OUTPUT_SIZE + i];
        }
        output[i] = sigmoid(output[i] + nn->output_biases[i]);
    }
}

// Backpropagation for a single sample
void backpropagation_sample(NeuralNetwork *nn, float *input, float *hidden, float *output, float *target,
                            float *ih_weights_delta, float *h_biases_delta,
                            float *ho_weights_delta, float *o_biases_delta) {
    float output_errors[OUTPUT_SIZE];
    float hidden_errors[HIDDEN_SIZE];

    // Calculate output layer errors
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output_errors[i] = (target[i] - output[i]) * sigmoid_derivative(output[i]);
    }

    // Accumulate hidden-output weights and output biases deltas
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            ho_weights_delta[i * OUTPUT_SIZE + j] += output_errors[j] * hidden[i];
        }
    }
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        o_biases_delta[i] += output_errors[i];
    }

    // Calculate hidden layer errors
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        hidden_errors[i] = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            hidden_errors[i] += output_errors[j] * nn->hidden_output_weights[i * OUTPUT_SIZE + j];
        }
        hidden_errors[i] *= sigmoid_derivative(hidden[i]);
    }

    // Accumulate input-hidden weights and hidden biases deltas
    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            ih_weights_delta[i * HIDDEN_SIZE + j] += hidden_errors[j] * input[i];
        }
    }
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        h_biases_delta[i] += hidden_errors[i];
    }
}

// Training function with mini-batch
void train(NeuralNetwork *nn, float *training_data, float *training_labels, int num_samples) {
    float hidden[HIDDEN_SIZE];
    float output[OUTPUT_SIZE];
    float target[OUTPUT_SIZE];

    float *ih_weights_delta = calloc(INPUT_SIZE * HIDDEN_SIZE, sizeof(float));
    float *h_biases_delta = calloc(HIDDEN_SIZE, sizeof(float));
    float *ho_weights_delta = calloc(HIDDEN_SIZE * OUTPUT_SIZE, sizeof(float));
    float *o_biases_delta = calloc(OUTPUT_SIZE, sizeof(float));

    int num_batches = num_samples / BATCH_SIZE;

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float total_loss = 0.0;

        for (int batch = 0; batch < num_batches; batch++) {
            memset(ih_weights_delta, 0, INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
            memset(h_biases_delta, 0, HIDDEN_SIZE * sizeof(float));
            memset(ho_weights_delta, 0, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
            memset(o_biases_delta, 0, OUTPUT_SIZE * sizeof(float));

            for (int i = 0; i < BATCH_SIZE; i++) {
                int idx = batch * BATCH_SIZE + i;
                
                // Prepare input and target
                float *input = &training_data[idx * INPUT_SIZE];
                int label = (int)training_labels[idx];
                memset(target, 0, OUTPUT_SIZE * sizeof(float));
                target[label] = 1.0;

                // Forward pass
                forward_pass(nn, input, hidden, output);

                // Calculate loss
                for (int j = 0; j < OUTPUT_SIZE; j++) {
                    total_loss += 0.5 * (target[j] - output[j]) * (target[j] - output[j]);
                }

                // Backpropagation
                backpropagation_sample(nn, input, hidden, output, target,
                                       ih_weights_delta, h_biases_delta,
                                       ho_weights_delta, o_biases_delta);
            }

            // Update weights and biases
            for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; i++) {
                nn->input_hidden_weights[i] += LEARNING_RATE * ih_weights_delta[i] / BATCH_SIZE;
            }
            for (int i = 0; i < HIDDEN_SIZE; i++) {
                nn->hidden_biases[i] += LEARNING_RATE * h_biases_delta[i] / BATCH_SIZE;
            }
            for (int i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; i++) {
                nn->hidden_output_weights[i] += LEARNING_RATE * ho_weights_delta[i] / BATCH_SIZE;
            }
            for (int i = 0; i < OUTPUT_SIZE; i++) {
                nn->output_biases[i] += LEARNING_RATE * o_biases_delta[i] / BATCH_SIZE;
            }
        }

        printf("Epoch %d, Loss: %f\n", epoch + 1, total_loss / num_samples);
    }

    free(ih_weights_delta);
    free(h_biases_delta);
    free(ho_weights_delta);
    free(o_biases_delta);
}

// Prediction function
int predict(NeuralNetwork *nn, float *input) {
    float hidden[HIDDEN_SIZE];
    float output[OUTPUT_SIZE];
    forward_pass(nn, input, hidden, output);
    
    int max_index = 0;
    for (int i = 1; i < OUTPUT_SIZE; i++) {
        if (output[i] > output[max_index]) {
            max_index = i;
        }
    }
    return max_index;
}

// Function to load MNIST data
void load_mnist(const char *filename, float *data, float *labels, int num_samples) {
    int fd = open(filename, O_RDONLY);
    if (fd == -1) {
        perror("Error opening file");
        exit(1);
    }

    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        perror("Error getting file size");
        exit(1);
    }

    void *file_memory = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (file_memory == MAP_FAILED) {
        perror("Error mapping file");
        exit(1);
    }

    unsigned char *file_data = (unsigned char *)file_memory;

    if (labels != NULL) {
        // Read labels
        file_data += 8;  // Skip header
        for (int i = 0; i < num_samples; i++) {
            labels[i] = (float)*file_data++;
        }
    } else if (data != NULL) {
        // Read images
        file_data += 16;  // Skip header
        for (int i = 0; i < num_samples * INPUT_SIZE; i++) {
            data[i] = *file_data++ / 255.0f;
        }
    }

    if (munmap(file_memory, sb.st_size) == -1) {
        perror("Error unmapping file");
        exit(1);
    }

    close(fd);
}

// Function to evaluate the model
float evaluate(NeuralNetwork *nn, float *test_data, float *test_labels, int num_samples) {
    int correct = 0;
    for (int i = 0; i < num_samples; i++) {
        int prediction = predict(nn, &test_data[i * INPUT_SIZE]);
        if (prediction == (int)test_labels[i]) {
            correct++;
        }
    }
    return (float)correct / num_samples;
}

// Main function
int main() {
    srand(time(NULL));

    NeuralNetwork nn;
    init_network(&nn);

    float *train_data = (float *)malloc(TRAIN_SIZE * INPUT_SIZE * sizeof(float));
    float *train_labels = (float *)malloc(TRAIN_SIZE * sizeof(float));
    float *test_data = (float *)malloc(TEST_SIZE * INPUT_SIZE * sizeof(float));
    float *test_labels = (float *)malloc(TEST_SIZE * sizeof(float));

    printf("Loading training data...\n");
    load_mnist("train-images.idx3-ubyte", train_data, NULL, TRAIN_SIZE);
    load_mnist("train-labels.idx1-ubyte", NULL, train_labels, TRAIN_SIZE);
    
    printf("Loading test data...\n");
    load_mnist("t10k-images.idx3-ubyte", test_data, NULL, TEST_SIZE);
    load_mnist("t10k-labels.idx1-ubyte", NULL, test_labels, TEST_SIZE);

    printf("Training started...\n");
    train(&nn, train_data, train_labels, TRAIN_SIZE);

    float accuracy = evaluate(&nn, test_data, test_labels, TEST_SIZE);
    printf("Test accuracy: %.2f%%\n", accuracy * 100);

    // Free allocated memory
    free(nn.input_hidden_weights);
    free(nn.hidden_biases);
    free(nn.hidden_output_weights);
    free(nn.output_biases);
    free(train_data);
    free(train_labels);
    free(test_data);
    free(test_labels);

    return 0;
}