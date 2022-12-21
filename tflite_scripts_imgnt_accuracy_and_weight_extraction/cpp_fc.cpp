#include <cstdint>
#include <iostream>
#include <fstream>
#include <dirent.h>
#include <chrono>

using namespace std::chrono;
using namespace std;

const int num_classes = 1000;
const int fc_layer_input_size = 1280;

const double weights_scale = 1.873968634754419327e-03;
const double ifms_scale = 0.020379824563860893;
const double biases_scale = 3.819115227088332176e-05;
int64_t ifm_zero_point = -128;

void read_ifms(string file_name,
               int8_t ifms[])
{
    int a;
    std::ifstream infile(file_name);
    bool failed = false;
    int line_num = 0;
    while (infile >> a)
    {
        ifms[line_num] = a;
        line_num++;
    }
}

void read_weights(string file_name,
                  int8_t fc_weights[])
{
    int a;
    std::ifstream infile(file_name);
    bool failed = false;
    int line_num = 0;
    while (infile >> a)
    {
        fc_weights[line_num] = a;
        line_num++;
    }
}

void read_weight_sums(string file_name,
                      int fc_weight_sums[])
{
    int a;
    std::ifstream infile(file_name);
    bool failed = false;
    int line_num = 0;
    while (infile >> a)
    {
        fc_weight_sums[line_num] = a;
        line_num++;
    }
}

void read_biases(string file_name,
                 int fc_biases[])
{
    int a;
    std::ifstream infile(file_name);
    bool failed = false;
    int line_num = 0;
    while (infile >> a)
    {
        fc_biases[line_num] = a;
        line_num++;
    }
}

void save_predictions(string file_name, string predictions)
{
    ofstream myfile;
    myfile.open(file_name);
    myfile << predictions;
    myfile.close();
}

string top_5_to_predictions_dict(int top5[5], string image_name)
{
    //{"dets": [527, 998, 879, 264, 403], "image": "ILSVRC2012_val_00018455.JPEG"}
    string dict = "{\"dets\": ["; // indices.tolist(), "image": file_names[i].replace('.txt', '.JPEG')}
    for (int i = 0; i < 5; i++)
    {
        dict += to_string(top5[i]);
        if (i < 4)
        {
            dict += ", ";
        }
    }
    dict += "], \"image\": \"" + image_name + "\"},";

    return dict;
}

void fc_layer(int8_t in_vector[], int8_t weights[], int weight_sums[], int top5[5], int biases[])
{
    double scaled_pss;
    double pss_vector[num_classes];
    top5[0] = -1;
    top5[1] = -1;
    top5[2] = -1;
    top5[3] = -1;
    top5[4] = -1;
    for (int i = 0; i < num_classes; i++)
    {
        int64_t pss = 0;
        const int row_start_index = i * fc_layer_input_size;
        for (int j = 0; j < fc_layer_input_size; j++)
        {
            pss += (int64_t)weights[row_start_index + j] * in_vector[j];
        }
        // if(i==999)cout << pss <<" "<<weight_sums[i]<<" "<<biases[i]<<"\n";
        pss_vector[i] = pss + (-weight_sums[i] * ifm_zero_point) + biases_scale * biases[i] / (weights_scale * ifms_scale);
    }
    for (int i = 0; i < 5; i++)
    {
        double max = - 1000000;
        for (int j = 0; j < num_classes; j++)
        {
            if(pss_vector[j] > max && top5[0] != j && top5[1] != j && top5[2] != j && top5[3] != j && top5[4] != j){
                max = pss_vector[j];
                top5[i] = j;
            }
        }
    }
}

int main()
{
    string weights_file =
        "/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/weights/fc_weights.txt";
    string weight_sums_file =
        "/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/weights/fc_weight_sums.txt";
    string biases_file =
        "/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/weights/fc_biases.txt";

    int8_t fc_weights[num_classes * fc_layer_input_size];
    int weight_sums[num_classes];
    int biases[num_classes];
    int8_t ifms[fc_layer_input_size];
    int top5[5];

    read_weights(weights_file, fc_weights);
    read_weight_sums(weight_sums_file, weight_sums);
    read_biases(biases_file, biases);

    string ifms_folder =
        "/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/fpga_out_1/";

    string predictions_file = "./predictions.json";

    string predictions_file_content = "[";

    DIR *dir;
    int img_count = 0;
    int images_to_test = -1;
    struct dirent *ent;
    if ((dir = opendir(ifms_folder.c_str())) != NULL)
    {
        /* print all the files and directories within directory */
        while ((ent = readdir(dir)) != NULL)
        {
            string file_path = ifms_folder + ent->d_name;
            string file_name = ent->d_name;
            if (file_path.find(".txt") == std::string::npos)
            {
                continue;
            }
            cout << file_path << "\n";
            read_ifms(file_path, ifms);
            img_count++;
            if (img_count == images_to_test)
            {
                break;
            }
            
            auto start = high_resolution_clock::now();
            fc_layer(ifms, fc_weights, weight_sums, top5, biases);
            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<microseconds>(stop - start);
            cout << duration.count() << endl;

            string formatted_file_name = file_name.substr(0, file_name.find(".", 0) + 1) + "JPEG";
            predictions_file_content += top_5_to_predictions_dict(top5, formatted_file_name);
        }
        closedir(dir);
    }
    else
    {
        return EXIT_FAILURE;
    }
    predictions_file_content = predictions_file_content.substr(0, predictions_file_content.length() - 1) + ']';
    save_predictions(predictions_file, predictions_file_content);
}