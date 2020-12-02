# DSC180A_Replication_Task-

### Project
Running LPA_GCN, GCN, and GraphSage on graph data. Notice that GCN is run through LPA_GCN by setting lambda to 0. 

run.py  
    --params:  
        model: the model chose to run. Now only support graph (GNN) and (LPA_GCN) Default is GCN
        image_path：The path of the image of drawing the input graph. If None, will not store a image.  
        dataset: the dataset. Now only support cora  
        cora_path: the path for cora dataset. Default is the relative path '/data/'  
        output_path: the path for output json file. Default is the relative path '/config/model-output.json'  
        hidden_neurons: How many hidden_neurons in one single hidden layer. Only needed when mode is GNN   
        len_walk: The length of random walk. Only needed when model is LPA_GNN   
        device: training device. Default is cuda  
        epochs: training epochs  
        lr: learning rate  
    more details try python run.py --help  
    
example:  
    >>> python run.py  
    >>> python run.py --hidden_neurons 100  
    >>> python run.py --hidden_neurons 100 --epochs 200    
    >>> python run.py --model LPA_GCN --Lambda 2
    >>> python run.py --model LPA_GCN --Lambda 2 --val_size 0.5
    
 
 ### reponsibility:
 * Xinrui Zhan: cora_loader.py, nlayer_gnn.py, packed functions into run.py
 * Yimei Zhao: dataloader.py, nlayer_gnn.py, draw.py, cora_loader.py
 
 second checkpoint:
 All: EDA functions, LPA_GCN
 ###