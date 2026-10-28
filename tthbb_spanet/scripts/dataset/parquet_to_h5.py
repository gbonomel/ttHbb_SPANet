import argparse

from tthbb_spanet.lib.dataset.spanet_dataset import SPANetDataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert awkward ntuples in coffea files to parquet files.')
    parser.add_argument('-c', '--cfg', type=str, required=True, help='YAML configuration file with input features')
    parser.add_argument('-i', '--input', type=str, required=True, nargs='+', help='Input parquet file')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output h5 file')
    parser.add_argument('-f', '--final_state', type=str, default='semileptonic', required=True, help='Select type of final state')
    parser.add_argument('-fm', '--fully_matched', action='store_true', help='Use only fully matched events')
    parser.add_argument('--no_shuffle', action='store_true', help='If set, do not shuffle the dataset')
    parser.add_argument('--reweigh', action='store_true', help='If set, scale event weights by a factor as specified in the configuration file')
    parser.add_argument('--entrystop', type=int, default=None, required=False, help='Number of events to process')
    parser.add_argument('--data', action='store_true', help='Input files are data: no parton matching targets, weights of 1 if missing')

    args = parser.parse_args()

    dataset = SPANetDataset(
        args.input,
        args.cfg,
        args.final_state,
        (not args.no_shuffle),
        args.reweigh,
        args.entrystop,
        args.data,
        args.fully_matched
    )
    dataset.save(args.output)
