import concurrent.futures
from collections import defaultdict

class NeuronDict:
    def __init__(self):
        self.dict = defaultdict(list)
    
    def add_item(self, workers):  
        for i in range(workers):
            post = 10  
            for j in range(5):
                self.dict[str(j)].append((str(post), 1))
                post += 1

    def add_item_concurrent(self, workers):
        post = 10

        def add(post):
            for j in range(5):
                self.dict[str(j)].append((str(post), 1))
                post += 1

        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(add, [post for i in range(workers)])
            for result in results:
                print(result)

def main():
    nd = NeuronDict()
    nd.add_item_concurrent(3)
    print(nd.dict.items())


if __name__ == "__main__":
    main()