#include "lenet.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define FILE_TRAIN_IMAGE		"train-images-idx3-ubyte"
#define FILE_TRAIN_LABEL		"train-labels-idx1-ubyte"
#define FILE_TEST_IMAGE		"t10k-images-idx3-ubyte"
#define FILE_TEST_LABEL		"t10k-labels-idx1-ubyte"
#define LENET_FILE 		"model_quant.dat"
#define COUNT_TRAIN		60000
#define COUNT_TEST		10000


int read_data(unsigned char(*data)[28][28], unsigned char label[], const int count, const char data_file[], const char label_file[])
{
    FILE *fp_image = fopen(data_file, "rb");
    FILE *fp_label = fopen(label_file, "rb");
    if (!fp_image||!fp_label) return 1;
	fseek(fp_image, 16, SEEK_SET);
	fseek(fp_label, 8, SEEK_SET);
	fread(data, sizeof(*data)*count, 1, fp_image);
	fread(label,count, 1, fp_label);
	fclose(fp_image);
	fclose(fp_label);
	return 0;
}

int print_weights_to_file(LeNet5 *lenet, const char *filename)
{
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        printf("Failed to open %s for writing weights\n", filename);
        return 1;
    }

    fprintf(fp, "=== LeNet5_quant Weights and Biases ===\n\n");

    // weight0_1: [1][6][5][5]
    fprintf(fp, "[weight0_1] (Input -> C1):\n");
    for (int o = 0; o < LAYER1; ++o) {
        fprintf(fp, "Filter %d:\n", o);
        for (int i = 0; i < LENGTH_KERNEL; ++i) {
            for (int j = 0; j < LENGTH_KERNEL; ++j) {
                fprintf(fp, "%d ", lenet->weight0_1[0][o][i][j]);
            }
            fprintf(fp, "\n");
        }
        fprintf(fp, "\n");
    }

    // bias0_1: [6]
    fprintf(fp, "[bias0_1] (C1):\n");
    for (int i = 0; i < LAYER1; ++i) {
        fprintf(fp, "%d ", lenet->bias0_1[i]);
    }
    fprintf(fp, "\n\n");

    // weight2_3: [6][16][5][5]
    fprintf(fp, "[weight2_3] (C2 -> C3):\n");
    for (int o = 0; o < LAYER3; ++o) {
        fprintf(fp, "Output Channel %d:\n", o);
        for (int i = 0; i < LAYER2; ++i) {
            fprintf(fp, "  Input Channel %d:\n", i);
            for (int y = 0; y < LENGTH_KERNEL; ++y) {
                for (int x = 0; x < LENGTH_KERNEL; ++x) {
                    fprintf(fp, "%d ", lenet->weight2_3[i][o][y][x]);
                }
                fprintf(fp, "\n");
            }
            fprintf(fp, "\n");
        }
    }

    // bias2_3: [16]
    fprintf(fp, "[bias2_3] (C3):\n");
    for (int i = 0; i < LAYER3; ++i) {
        fprintf(fp, "%d ", lenet->bias2_3[i]);
    }
    fprintf(fp, "\n\n");

    // weight4_5: [16][120][5][5]
    fprintf(fp, "[weight4_5] (C4 -> FC1):\n");
    for (int o = 0; o < LAYER5; ++o) {
        fprintf(fp, "FC Neuron %d:\n", o);
        for (int i = 0; i < LAYER4; ++i) {
            fprintf(fp, "  From Map %d:\n", i);
            for (int y = 0; y < LENGTH_KERNEL; ++y) {
                for (int x = 0; x < LENGTH_KERNEL; ++x) {
                    fprintf(fp, "%d ", lenet->weight4_5[i][o][y][x]);
                }
                fprintf(fp, "\n");
            }
            fprintf(fp, "\n");
        }
    }

    // bias4_5: [120]
    fprintf(fp, "[bias4_5] (FC1):\n");
    for (int i = 0; i < LAYER5; ++i) {
        fprintf(fp, "%d ", lenet->bias4_5[i]);
    }
    fprintf(fp, "\n\n");

    // weight5_6: [120 * FEATURE5_SIZE][10]
    fprintf(fp, "[weight5_6] (FC1 -> Output):\n");
    int flattened_input_size = LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5;
    for (int o = 0; o < OUTPUT; ++o) {
        fprintf(fp, "Output Neuron %d:\n", o);
        for (int i = 0; i < flattened_input_size; ++i) {
            fprintf(fp, "%d ", lenet->weight5_6[i][o]);
            if ((i + 1) % 20 == 0) fprintf(fp, "\n"); // wrap for readability
        }
        fprintf(fp, "\n\n");
    }

    // bias5_6: [10]
    fprintf(fp, "[bias5_6] (Output):\n");
    for (int i = 0; i < OUTPUT; ++i) {
        fprintf(fp, "%d ", lenet->bias5_6[i]);
    }
    fprintf(fp, "\n\n");

    fclose(fp);
    printf("Weights and biases saved to %s\n", filename);
    return 0;
}

/*void training(LeNet5 *lenet, image *train_data, uint8 *train_label, int batch_size, int total_size)
{
	for (int i = 0, percent = 0; i <= total_size - batch_size; i += batch_size)
	{
		TrainBatch(lenet, train_data + i, train_label + i, batch_size);
		if (i * 100 / total_size > percent)
			printf("batchsize:%d\ttrain:%2d%%\n", batch_size, percent = i * 100 / total_size);
	}
}*/

int testing(LeNet5 *lenet, image *test_data, uint8 *test_label,int total_size)
{
	int right = 0, percent = 0;
	for (int i = 0; i < total_size; ++i)
	{
		uint8 l = test_label[i];
		int p = Predict(lenet, test_data[i], 10);
		right += l == p;
		if (i * 100 / total_size > percent)
			printf("test:%2d%%\n", percent = i * 100 / total_size);
	}
	return right;
}

int save(LeNet5 *lenet, char filename[])
{
	FILE *fp = fopen(filename, "wb");
	if (!fp) return 1;
	fwrite(lenet, sizeof(LeNet5), 1, fp);
	fclose(fp);
	return 0;
}

int load(LeNet5 *lenet, char filename[])
{
	FILE *fp = fopen(filename, "rb");
	if (!fp) return 1;
	fread(lenet, sizeof(LeNet5), 1, fp);
	fclose(fp);
	return 0;
}

void save_weights(LeNet5 *lenet, const char *filename)
{
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        printf("Error opening file to save weights!\n");
        return;
    }

    fprintf(fp, "--- Saving LeNet5 Weights ---\n");

    // weight0_1: input -> layer1 (conv1)
    fprintf(fp, "\nweight0_1 (input -> layer1):\n");
    for (int i = 0; i < INPUT; ++i) {
        for (int j = 0; j < LAYER1; ++j) {
            for (int x = 0; x < LENGTH_KERNEL; ++x) {
                for (int y = 0; y < LENGTH_KERNEL; ++y) {
                    fprintf(fp, "%d ", lenet->weight0_1[i][j][x][y]);
                }
                fprintf(fp, "\n");
            }
            fprintf(fp, "\n");
        }
        fprintf(fp, "\n");
    }

    // weight2_3: layer2 -> layer3 (conv3)
    fprintf(fp, "\nweight2_3 (layer2 -> layer3):\n");
    for (int i = 0; i < LAYER2; ++i) {
        for (int j = 0; j < LAYER3; ++j) {
            for (int x = 0; x < LENGTH_KERNEL; ++x) {
                for (int y = 0; y < LENGTH_KERNEL; ++y) {
                    fprintf(fp, "%d ", lenet->weight2_3[i][j][x][y]);
                }
                fprintf(fp, "\n");
            }
            fprintf(fp, "\n");
        }
        fprintf(fp, "\n");
    }

    // weight4_5: layer4 -> layer5 (conv5)
    fprintf(fp, "\nweight4_5 (layer4 -> layer5):\n");
    for (int i = 0; i < LAYER4; ++i) {
        for (int j = 0; j < LAYER5; ++j) {
            for (int x = 0; x < LENGTH_KERNEL; ++x) {
                for (int y = 0; y < LENGTH_KERNEL; ++y) {
                    fprintf(fp, "%d ", lenet->weight4_5[i][j][x][y]);
                }
                fprintf(fp, "\n");
            }
            fprintf(fp, "\n");
        }
        fprintf(fp, "\n");
    }

    // weight5_6: fully connected (flattened layer5 -> output)
    fprintf(fp, "\nweight5_6 (flattened layer5 -> output):\n");
    for (int i = 0; i < LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5; ++i) {
        for (int j = 0; j < OUTPUT; ++j) {
            fprintf(fp, "%d ", lenet->weight5_6[i][j]);
        }
        fprintf(fp, "\n");
    }

    // Biases
    fprintf(fp, "\nBiases:\n");

    fprintf(fp, "\nbias0_1 (layer1 biases):\n");
    for (int i = 0; i < LAYER1; ++i) {
        fprintf(fp, "%d ", lenet->bias0_1[i]);
    }
    fprintf(fp, "\n");

    fprintf(fp, "\nbias2_3 (layer3 biases):\n");
    for (int i = 0; i < LAYER3; ++i) {
        fprintf(fp, "%d ", lenet->bias2_3[i]);
    }
    fprintf(fp, "\n");

    fprintf(fp, "\nbias4_5 (layer5 biases):\n");
    for (int i = 0; i < LAYER5; ++i) {
        fprintf(fp, "%d ", lenet->bias4_5[i]);
    }
    fprintf(fp, "\n");

    fprintf(fp, "\nbias5_6 (output biases):\n");
    for (int i = 0; i < OUTPUT; ++i) {
        fprintf(fp, "%d ", lenet->bias5_6[i]);
    }
    fprintf(fp, "\n");

    fclose(fp);
}


void foo()
{
	image *train_data = (image *)calloc(COUNT_TRAIN, sizeof(image));
	uint8 *train_label = (uint8 *)calloc(COUNT_TRAIN, sizeof(uint8));
	image *test_data = (image *)calloc(COUNT_TEST, sizeof(image));
	uint8 *test_label = (uint8 *)calloc(COUNT_TEST, sizeof(uint8));
	if (read_data(train_data, train_label, COUNT_TRAIN, FILE_TRAIN_IMAGE, FILE_TRAIN_LABEL))
	{
		printf("ERROR!!!\nDataset File Not Find!Please Copy Dataset to the Floder Included the exe\n");
		free(train_data);
		free(train_label);
		system("pause");
	}
	if (read_data(test_data, test_label, COUNT_TEST, FILE_TEST_IMAGE, FILE_TEST_LABEL))
	{
		printf("ERROR!!!\nDataset File Not Find!Please Copy Dataset to the Floder Included the exe\n");
		free(test_data);
		free(test_label);
		system("pause");
	}


	LeNet5 *lenet = (LeNet5 *)malloc(sizeof(LeNet5));
	if (load(lenet, LENET_FILE))
		Initial(lenet);

    print_weights_to_file(lenet, "weights_dump.txt");

	clock_t start = clock();
	int batches[] = { 300 };
    /*
	for (int i = 0; i < sizeof(batches) / sizeof(*batches);++i)
		training(lenet, train_data, train_label, batches[i],COUNT_TRAIN);
        */
	int right = testing(lenet, test_data, test_label, COUNT_TEST);
	printf("%d/%d\n", right, COUNT_TEST);
	printf("Time:%u\n", (unsigned)(clock() - start));
	save_weights(lenet, "lenet_weights.txt");
	save(lenet, LENET_FILE);
	free(lenet);
	free(train_data);
	free(train_label);
	free(test_data);
	free(test_label);
	system("pause");
}

int main()
{
	foo();
	return 0;
}