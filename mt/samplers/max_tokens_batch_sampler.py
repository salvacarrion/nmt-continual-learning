import random
from torch.utils.data.sampler import BatchSampler, RandomSampler, SubsetRandomSampler
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

        # Not a clean solution
        self.bucket_batches = []
        self._build_buckets()

    def __iter__(self):
        if self.shuffle:  # Shuffle everything
            self._build_buckets()

        # Iterate over buckets
        for batches, batch_sizes in self.bucket_batches:
            # Shuffle bucket-batch order
            batches = SubsetRandomSampler(batches) if self.shuffle else batches
            for batch in batches:
                if self.shuffle:  # Shuffle inner batch
                    random.shuffle(batch)
                yield batch  # Batch indexes [sent1_idx, sent2_idx,...]

    def __len__(self):
        return sum([len(x[0]) for x in self.bucket_batches])

    def _build_buckets(self):
        # Randomize samples
        tmp_sampler = RandomSampler(self.sampler) if self.shuffle else self.sampler

        # Split samples in N batches (or "buckets")
        tmp_sampler = BatchSampler(tmp_sampler, min(self.batch_size * self.bucket_size_multiplier, len(self.sampler)),
                                   False)

        # Sort samples
        self.bucket_batches = []
        for bucket in tmp_sampler:
            bucket_sorted = sorted([(i, self.sort_key(i)) for i in bucket], key=lambda x: x[1])

            # Create batches constrained
            batches = []
            batch_sizes = []

            last_batch = []
            last_batch_size = 0
            for i, (sample_i, length_i) in enumerate(bucket_sorted):
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
            self.bucket_batches.append((batches, batch_sizes))
