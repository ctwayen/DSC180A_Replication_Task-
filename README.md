# DSC180A_Replication_Task-

run.py
    --params:
        model: the model chose to run. Now only support graph (GNN)
        layer_number： how many hidden layers in the GNN
        dataset: the dataset. Now only support cora
        cora_path: the path for cora dataset. Default is the relative path '/data/'
        output_path: the path for output json file. Default is the relative path '/config/model-output.json'
        channels: layers channel output for each layer
        dropout: dropout ratio for dropout layer
        epochs: training epochs
        lr: learning rate
    more details try python run.py --help
    
example:
    >>> python run.py
    >>> python run.py --layer_number 3
    >>> python run.py --layer_number 1 --epochs 250
    