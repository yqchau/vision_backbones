echo Running example of benchmarking memory requirements...
for i in {0..592}
    do
    python benchmark_memory.py --i $i;
done
