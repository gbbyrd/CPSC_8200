import csv
import glob
import matplotlib.pyplot as plt

# get all of the .csv file paths
csv_paths = glob.glob('./*.csv')

benchmarking_results = dict()

for csv_path in csv_paths:
    with open(csv_path, 'r') as file:
        csvreader = csv.reader(file)
    
        gflops = []
        time = []
        operations = []

        for row in csvreader:
            row = row[0].split(';')
            gflops.append(float(row[0]))
            time.append(float(row[1]))
            operations.append(float(row[2]))

    benchmarking_results[csv_path[:-14]] = {
        'gflops': gflops,
        'time': time,
        'operations': operations
    }

plt.figure(figsize=(10, 10))

for key in benchmarking_results:
    data = benchmarking_results[key]
    plt.plot(data['operations'], data['gflops'], label=key)

plt.legend()
plt.xlabel('Num Operations (10^5)')
plt.ylabel('Performance (GFLOPS)')
# plt.show()
plt.savefig('performance.png')
