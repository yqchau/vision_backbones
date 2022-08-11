echo Running example of overwriting batch size...
for bs in 64 128 256 512 1024 2048 4096
    do
    echo batch_size: $bs;
    python train.py datamodule.batch_size=$bs;
done
