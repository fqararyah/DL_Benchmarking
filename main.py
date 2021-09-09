from BenchmarkModel import BenchmarkModel
import utils
import BenchmarkModel
import Settings

benchmark_models = utils.raed_benchmarks()

with open (Settings.Settings().status_file_name, 'w') as f:
    f.write('start')

for model in benchmark_models:
    model.get_metrics()

with open (Settings.Settings().status_file_name, 'w') as f:
    f.write('stop')