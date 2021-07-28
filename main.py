from BenchmarkModel import BenchmarkModel
import utils
import BenchmarkModel

benchmark_model = utils.raed_benchmarks()

for model in benchmark_model:
    model.get_metrics()