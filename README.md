# DSC180A_Replication_Task-

run.py  
    --params:
        model: the model chose to run. Now only support graph (GNN)  
        image_path：The path of the image of drawing the input graph. If None, will not store a image.  
        dataset: the dataset. Now only support cora  
        cora_path: the path for cora dataset. Default is the relative path '/data/'  
        output_path: the path for output json file. Default is the relative path '/config/model-output.json'  
        hidden_neurons: How many hidden_neurons in one single hidden layer  
        device: training device. Default is cuda  
        epochs: training epochs  
        lr: learning rate  
    more details try python run.py --help  
    
example:  
    >>> python run.py  
    >>> python run.py --hidden_neurons 100  
    >>> python run.py --hidden_neurons 100 --epochs 200   
    
 
 ### reponsibility:
 * Xinrui Zhan: cora_loader.py, nlayer_gnn.py, packed functions into run.py
 * Yimei Zhao: dataloader.py, nlayer_gnn.py, draw.py, cora_loader.py
 
 second checkpoint:
 All: EDA functions
 ###