import numpy as np

class CountingBloomFilter():
    """
    Implement the counting bloom filter (CBF) which supports: 
    - search: queries the membership of an element
    - insert: inserts a string to the filter
    - delete: removes a string from the filter
    """

    def __init__(self, num_items, fpr=0.02, num_hash_functions = 0):
        """
        Initialize the CBF with a specified fpr
        for a number of items

        Parameters
        ----------
        - num_items: int, expected number of items
        - fpr: float, false-positive rate
        - num_hash_functions: int, number of hash functions to use

        Returns
        -------
        None
        """
        self.fpr = fpr if fpr > 0 else 0.02

        # The CBF is initially empty
        self.num_items = 0
        self.memory_size = int(num_items * (-1.44 * np.log2(fpr)))

        # Initialize the hash table
        self.hash_table = [0] * self.memory_size

        # Initialize an optimal number of hash 
        # functions unless specified otherwise
        if num_hash_functions == 0:
            self.num_hash_functions = int(round(-np.log2(fpr))) \
                if int(-np.log2(fpr)) > 0 else 1
        else:
            self.num_hash_functions = num_hash_functions


    def str_to_int(self, item, i):
        """
        Convert a string to an integer 
        representation for hashing

        Parameters
        ----------
        - item: str, the string to convert
    
        - i: int, the i-th hash function we're 
                  calculating for

        Returns
        -------
        - int: the integer value to be used
               for the final hash
        """
        # An arbitrary base
        d = 2 ** (i + 1)
        m = self.memory_size

        # Calculate the item's integer
        # representation based on each character
        final_num = 0
        str_length = len(item)
        for char_index in range(str_length):
            int_char = ord(item[char_index]) * (d ** char_index)
            final_num = (final_num + (int_char ** 2)) % m

        return final_num

    def hash_cbf(self, item):
        """
        Calculates the hash indices of an item

        Parameters
        ----------
        - item: str, the string to hash

        Returns
        -------
        - list: the hash table indices
                for the item
        """
        # All hash values of the item
        hash_indices = []

        # Constants for our hash function
        k = self.num_hash_functions

        # Pass item through every hash function
        for i in range(k):
            # Produce the item's hash
            # based on each character
            hash_index = self.str_to_int(item, i)
            hash_indices.append(hash_index)

        return hash_indices


    def search(self, item):
        """
        Check if an item (represented by
        a string) exists in the CBF

        Parameters
        ----------
        - item: str, the string to find in the CBF

        Returns
        -------
        - boolean: whether item exists in the CBF
        """
        # Hash the item
        hash_indices = self.hash_cbf(item)

        # Check for indices with zero counters
        return self.search_hashes(hash_indices)


    def search_hashes(self, hash_indices):
        """
        Check if an item (represented by a 
        set of hash indices) exists in the CBF

        Parameters
        ----------
        - hash_indices: list, the list of hash
                        indices to check in the CBF

        Returns
        -------
        - boolean: whether item exists in the CBF
        """
        # Check if any of the counters
        # at the item's indices are zero
        for hash_index in hash_indices:
            if self.hash_table[hash_index] == 0:
                return False

        # All counters are more than zero
        # —the item likely exists
        return True

    def insert(self, item):
        """
        Insert an item into the CBF

        Parameters
        ----------
        - item: str, the string to insert

        Returns
        -------
        None
        """
        # Hash the item
        hash_indices = self.hash_cbf(item)

        # Increment the counter at the
        # item's indices by 1
        for hash_index in hash_indices:
            self.hash_table[hash_index] += 1

        # Increment the total number of 
        # items in our CBF
        self.num_items += 1


    def delete(self, item):
        """
        Delete an item from the CBF

        Parameters
        ----------
        - item: str, the string to delete

        Returns
        -------
        None
        """
        # Hash the item
        hash_indices = self.hash_cbf(item)

        # Decrement the counter at the item's 
        # indices by 1 if the item exists
        if self.search_hashes(hash_indices):
            for hash_index in hash_indices:
                self.hash_table[hash_index] -= 1

        # Decrement the total number of 
        # items in our CBF
        self.num_items -= 1
