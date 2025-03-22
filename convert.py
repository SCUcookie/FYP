class precision_context:
    def __init__(self, model, dtype):
        self.model = model
        self.dtype = dtype
        self.original_dtypes = {}
        
    def __enter__(self):
        # 记录并转换精度
        for name, param in self.model.named_parameters():
            self.original_dtypes[name] = param.dtype
            param.data = param.data.to(self.dtype)
        return self.model
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # 恢复原始精度
        for name, param in self.model.named_parameters():
            param.data = param.data.to(self.original_dtypes[name])