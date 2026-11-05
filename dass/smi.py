import submodlib
import submodlib.functions as submod_fn

SMI_METHODS = ("fl1mi", "fl2mi", "gcmi", "logdetmi")
TARGET_METHODS = ("fl", "gc", "logdet")


class SMI():
    def __init__(
            self, target_domain_embeddings, source_domain_embeddings, 
            optimizer:str="LazyGreedy", method:str="fl1mi", metric:str="cosine",
            eta:float=1.0, nu:float=1.0, lambdaVal:float=1.0, verbose:bool=False,
            stopIfZeroGain:bool=False, stopIfNegativeGain:bool=False 
        ):
        self.nu=nu
        self.eta=eta
        self.method=method
        self.metric=metric
        self.verbose=verbose
        self.optimizer=optimizer
        self.lambdaVal=lambdaVal
        self.stopIfZeroGain=stopIfZeroGain
        self.stopIfNegativeGain=stopIfNegativeGain
        self.target_domain=target_domain_embeddings
        self.source_domain=source_domain_embeddings
        if method not in SMI_METHODS:
            raise ValueError(f"Unknown SMI method '{method}'. Expected one of: {SMI_METHODS}")

    def select(self, budget):
        
        # Compute source kernel
        data_sijs = submodlib.helper.create_kernel(
            X=self.source_domain, metric=self.metric, method="sklearn")
        
        #Compute target-target kernel
        if(self.method=='logdetmi'):
            query_query_sijs = submodlib.helper.create_kernel(
                X=self.target_domain, metric=self.metric, method="sklearn")
        
        #Compute source-target kernel
        query_sijs = submodlib.helper.create_kernel(
            X=self.target_domain, X_rep=self.source_domain,
            metric=self.metric, method="sklearn")

        if(self.method=='fl1mi'):
            obj = submodlib.FacilityLocationMutualInformationFunction(
                n=self.source_domain.shape[0], num_queries=self.target_domain.shape[0],
                data_sijs=data_sijs, query_sijs=query_sijs, magnificationEta=self.eta)

        if(self.method=='fl2mi'):
            obj = submodlib.FacilityLocationVariantMutualInformationFunction(
                n=self.source_domain.shape[0], num_queries=self.target_domain.shape[0],
                query_sijs=query_sijs, queryDiversityEta=self.eta)
        
        if(self.method=='gcmi'):
            obj = submodlib.GraphCutMutualInformationFunction(
                n=self.source_domain.shape[0], num_queries=self.target_domain.shape[0],
                query_sijs=query_sijs, metric=self.metric)
            
        if(self.method=='logdetmi'):
            obj = submodlib.LogDeterminantMutualInformationFunction(
                n=self.source_domain.shape[0], num_queries=self.target_domain.shape[0],
                data_sijs=data_sijs, query_sijs=query_sijs, query_query_sijs=query_query_sijs,
                magnificationEta=self.eta, lambdaVal=self.lambdaVal)

        greedyList = obj.maximize(budget=budget,optimizer=self.optimizer, stopIfZeroGain=self.stopIfZeroGain, 
                              stopIfNegativeGain=self.stopIfNegativeGain, verbose=self.verbose)
        greedyIndices = [x[0] for x in greedyList]
        return greedyIndices
    

class SubmodSelection():
    def __init__(self, target_domain_embeddings, method:str="fl", optimizer:str="LazyGreedy"):
        self.embeddings=target_domain_embeddings
        self.method=method
        self.optimizer=optimizer
        if method not in TARGET_METHODS:
            raise ValueError(f"Unknown target selection method '{method}'. Expected one of: {TARGET_METHODS}")
    
    def select(self, budget):
        data_sijs=submodlib.helper.create_kernel(X=self.embeddings, metric="cosine", method="sklearn")
        if self.method=="fl":
            print("Instantiating facility location function...")
            submod_obj=submod_fn.facilityLocation.FacilityLocationFunction(
                n=self.embeddings.shape[0], mode="dense", sijs=data_sijs, separate_rep=False)
        elif self.method=="gc":
            print("Instantiating graph cut function...")
            submod_obj=submod_fn.graphCut.GraphCutFunction(
                n=self.embeddings.shape[0], mode="dense", ggsijs=data_sijs, lambdaVal=0.4, separate_rep=False)
        elif self.method=="logdet":
            print("Instantiating log-determinant function...")
            submod_obj=submod_fn.logDeterminant.LogDeterminantFunction(
                n=self.embeddings.shape[0], mode="dense", sijs=data_sijs, lambdaVal=1)

        print("Running the lazy greedy algorithm...")
        greedyList=submod_obj.maximize(budget=budget, optimizer=self.optimizer, show_progress=True)
        greedyIndices = [x[0] for x in greedyList]
        return greedyIndices