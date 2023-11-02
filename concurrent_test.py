from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager

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
        updates = [[(str(i), i) for i in range(1000)],
                   [(str(i), i) for i in range(1000, 2000)],
                   [(str(i), i) for i in range(2000, 3000)],
        ]

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

        # Print the updated dictionary
        print(dict(shared_dict))

if __name__ == '__main__':
    main()
