import math

from torch.utils.data.sampler import BatchSampler, RandomSampler, SequentialSampler, SubsetRandomSampler

from torchnlp.samplers.sorted_sampler import SortedSampler
from torchnlp.utils import identity


class MaxTokensBatchSampler(BatchSampler):

    def __init__(self,
                 sampler,
                 batch_size,
                 max_tokens,
                 drop_last,
                 sort_key=identity,
                 bucket_size_multiplier=100,
                 shuffle=True):
        super().__init__(sampler, batch_size, drop_last)
        self.max_tokens = max_tokens
        self.sort_key = sort_key
        self.bucket_size_multiplier = bucket_size_multiplier
        self.shuffle = shuffle

    def __iter__(self):
        # Randomize samples
        rnd_sampler = RandomSampler(self.sampler) if self.shuffle else self.sampler

        # Split samples in N batches (or "buckets")
        bucket_sampler = BatchSampler(rnd_sampler, min(self.batch_size * self.bucket_size_multiplier, len(self.sampler)), False)

        # Sort samples
        bucket_batches = []
        for bucket in bucket_sampler:
            bucket_lengths = sorted([(i, self.sort_key(i)) for i in bucket], key=lambda x: x[1])

            # Create batches constrained
            batches = []
            batch_sizes = []

            last_batch = []
            last_batch_size = 0
            for i, (sample_i, length_i) in enumerate(bucket_lengths):
                if (last_batch_size + length_i) < self.max_tokens:
                    last_batch.append(sample_i)
                    last_batch_size += length_i
                else:
                    # Add batch
                    batches.append(last_batch)
                    batch_sizes.append(last_batch_size)

                    # Add new sample
                    last_batch = [sample_i]
                    last_batch_size = length_i

            # Add last batch
            batches.append(last_batch)
            batch_sizes.append(last_batch_size)

            # Add bucket batches
            bucket_batches.append((batches, batch_sizes))

        for batches, batch_sizes in bucket_batches:
            for batch in batches:
                yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return math.ceil(len(self.sampler) / self.batch_size)
