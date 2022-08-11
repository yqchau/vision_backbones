for opt in SGD Adam AdamW Nadam Radam AdamP SGDP Adadelta Adafactor RMSprop NovoGrad
    do
        echo Checking Optimizer $opt;
        python train.py optimizers.option=$opt logger.experiment_version_tag=$opt logger.experiment_name=optimizers_check;
done
