# DSC180A_Replication_Task-

### Project
Running LPA_GCN, GCN, and GraphSage on graph data. Notice that GCN is run through LPA_GCN by setting lambda to 0.   

run.py  
    --params:  
        model: the model chose to run. Only support graph (GCN), (LPA_GCN), (n_GCN). Default is GCN      
        dataset: the dataset. Only support cora and arxiv   
        cora_path: the path for cora dataset. Default is the relative path '/data/'    
        output_path: the path for output json file. Default is the relative path '/config/model-output.json'   
        n: how many hidden layers used in n_GCN default is 0   
        self_weight: the weight of self loop used in n_GCN default is 10   
        val_size: the proportion of validation dataset default is 0.3   
        agg_func: the aggregate function used in graphsage default is MEAN   
        num_neigh: How many neighbors we used in graphsage default is 10. Be carefull using this since some number may cause errors (do not have such amount nodes)   
        hidden_neurons: How many hidden_neurons in one single hidden layer. Only needed when mode is GNN     
        len_walk: The length of random walk. Only needed when model is LPA_GNN     
        device: training device. Default is cuda    
        epochs: training epochs    
        lr: learning rate    
    more details try python run.py --help    
    
cora example:
    running GCN try:  
    >>> python run.py    
    >>> python run.py --hidden_neurons 100    
    >>> python run.py --hidden_neurons 100 --epochs 200    
    >>> python run.py --hidden_neurons 100 --epochs 200 --val_size 0.5  
    running n_GCN try:   
    >>> python run.py --model n_GCN   
    >>> python run.py --model n_GCN --n 1 --self_weight 20     
    running LPA_GCN try:  
    >>> python run.py --model LPA_GCN --Lambda 2   
    >>> python run.py --model LPA_GCN --Lambda 2 --val_size 0.5    
    running graphsage try:     
    >>> python run.py --model graphsage    
    >>> python run.py --model graphsage --num_neigh 5 --agg_func MAX    
    
arxiv example:   
    >>> python run.py --dataset arxiv    
    >>> python run.py --dataset arxiv --arxiv_size 0.01  
    >>> python run.py --dataset arxiv --arxiv_size 0.01 --seed 10  
All the parameters about models run in cora could also work with arxiv

#### Be careful with choosing the value of arxiv_size. The limitation of  datahub is approximatly aournd 0.05 of the whole dataset. Running models on the whole dataset (1) approximatly need >1tb storage. 
 
 ### reponsibility:
 Xinrui Zhan: run.py, LPA_GCN.py, n_GCN.py, GraphSage.py, coraloader.py, arxiv_loader.py 
 ShangLi: data/edges.csv, data/labels.csv
 Yimei Zhao: draw.py, eda_functions.py
 ### Links:  
 LPA_GCN paper: https://arxiv.org/pdf/2002.06755.pdf  
 GraphSage paper: https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf  
