from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
import time

def update_dict(shared_dict, input):
    # Update the shared dictionary
    for key, value in input:
        shared_dict[key] = value

def main():
    # Create a manager object
    with Manager() as manager:
        # Create a shared dictionary
        shared_dict = manager.dict()

        # Initialize shared dictionary with some values
        shared_dict['initial_key'] = 'initial_value'

        # List of keys and values to update the dictionary with
        updates = [[(str(i), i) for i in range(100000)],
                   [(str(i), i) for i in range(100000, 200000)],
                   [(str(i), i) for i in range(200000, 300000)],
        ]

        start = time.time()
        # Use a ProcessPoolExecutor to update the shared dictionary
        with ProcessPoolExecutor() as executor:
            futures = []
            for update in updates:
                # Submit the update_dict function to the executor
                future = executor.submit(update_dict, shared_dict, update)
                futures.append(future)

            # Wait for all submitted tasks to be completed
            for future in futures:
                future.result()

        end = time.time()
        # Print the updated dictionary
        dict(shared_dict)
        print(end-start)

def compare():

    start = time.time()
    updates = [[(str(i), i) for i in range(100000)],
               [(str(i), i) for i in range(100000, 200000)],
               [(str(i), i) for i in range(200000, 300000)],
        ]
    shared_dict = {}

    for l in updates:
        update_dict(shared_dict, l)

    end = time.time()

    print(end-start)
    


if __name__ == '__main__':
    main()
