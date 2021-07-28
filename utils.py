from BenchmarkModel import BenchmarkModel
import Settings
import BenchmarkModel

settings = Settings.Settings()

def raed_benchmarks():
    benchmark_models = []
    with open(settings.networks_file, 'r') as f:
        for line in f:
            if line == settings.end_of_file:
                break

            benchmark_models.append(BenchmarkModel(model_name= line.replace(' ', '')))

    models_batch_sizes_dict = {}
    with open(settings.batch_sizes_file, 'r') as f:
        for line in f:
            if line == settings.end_of_file:
                break
            splits = line.replace(' ', '').split(settings.delimiter)
            model_name = splits[0]
            models_batch_sizes_dict[model_name] = []
            batch_sizes = splits[1].replace('[', '').replace(']', '')
            if ':' in batch_sizes:
                range_beginning = int(batch_sizes.split(':')[0])
                range_end = int(batch_sizes.split(':')[1])
                while range_beginning < range_end:
                    models_batch_sizes_dict[model_name].append(range_beginning)
                    range_beginning *= 2
            else:
                for split in batch_sizes.split(','):
                    models_batch_sizes_dict[model_name].append(int(split))
            
    for model in benchmark_models:
        if model.model_name in models_batch_sizes_dict:
            model.batch_sizes = models_batch_sizes_dict[model.model_name]
        else:
            model.batch_sizes = models_batch_sizes_dict[settings.global_setting_keyword]
    
    return benchmark_models


