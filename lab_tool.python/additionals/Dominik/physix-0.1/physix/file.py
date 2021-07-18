class Data:
    data = None
    keys = None
    
    def __init__(self, data, keys=None):
        self.data = data
        
        if keys is None:
            self.keys = [str(index) for index in range(len(self.data))]
        else:
            self.keys = [str(element) for element in keys]
        
        if self.data.shape[0] != len(self.keys):
            raise Exception("number of keys not matching data")
        
    def __getitem__(self, key):
        if not str(key) in self.keys:
            raise Exception("access of non-existent column key")
        
        return self.data[self.keys.index(str(key))]
        
    def __getattr__(self, key):
        return self[key]

def load(path, columns=None, delimiter=None, skip=0):
    from numpy import loadtxt
    
    data = loadtxt(path, usecols=columns, delimiter=delimiter, skiprows=skip, ndmin=2, unpack=True)
    keys = None
    
    if isinstance(columns, dict):
        keys = [str(value) for value in columns.values()]
    elif isinstance(columns, list):
        keys = [str(element) for element in columns]
    
    return Data(data, keys)
