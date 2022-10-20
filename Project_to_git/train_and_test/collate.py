import torch



class PadSequence:
    def __call__(self, batch):
        # Each element in "batch" is a tuple (data, label, file)
        # Sort the batch in the descending order
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
        # Get each sequence and pad it
        sequences = [x[0] for x in sorted_batch]
        sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        # Also need to store the length of each sequence
        # This is later needed in order to unpad the sequences
        lengths = torch.LongTensor([len(x) for x in sequences])
        # Get the labels and files of the *sorted* batch
        labels = torch.LongTensor([x[1] for x in sorted_batch])
        original_files = [x[2] for x in sorted_batch]
        return sequences_padded, labels, original_files, lengths
