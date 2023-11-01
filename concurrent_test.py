import concurrent.futures
from functools import partial

lst = []

def add(number):
    return [n for n in range(number)]

def main():
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(add, range(3))
        for result in results:
            lst.extend(result)
    print(lst)

if __name__ == "__main__":
    main()