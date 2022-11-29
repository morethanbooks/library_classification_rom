
import os
import pandas as pd


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./data/concat_2000_2019_data')
    parser.add_argument('--format', default='tsv')
    parser.add_argument('--output-format', default='tsv')
    parser.add_argument('--label-hierarchy', default='./../data/label-hierarchy.tsv', 
        help='This file is generated by hierarchy.py')
    args = parser.parse_args()

    if args.format == "tsv":
        df = pd.read_csv(args.data + "." + args.format, sep='\t')
    elif args.format == "parquet":
        df = pd.read_parquet(args.data + "." + args.format)

    hierarchy = pd.read_csv(args.label_hierarchy, sep='\t')

    # filter unlabeled instances
    labeled = df[~pd.isna(df['BK_notation'])]

    # filter unknown labels
    labeled['BK_split'] = labeled['BK_notation'].apply(lambda row: row.split("|"))
    labels = set(hierarchy['node'])
    index = [i for i in labeled.index if all(l in labels for l in labeled.loc[i]['BK_split'])]
    labeled = labeled.loc[index]

    # splits
    train = labeled.sample(frac=0.8, random_state=1001)
    test = labeled.drop(train.index)
    dev = test.sample(frac=0.5, random_state=1001)
    test = test.drop(dev.index)

    # serialize
    def get_name(split):
        basename = os.path.basename(args.data)
        rest = [basename]
        if "." in basename:
            # drop extension
            *rest, _ = basename.split('.')
        return ".".join(rest + [split] + [args.output_format])

    outdir = os.path.dirname(args.data)

    if args.output_format == "tsv":
        train.to_csv(os.path.join(outdir, get_name('train')))
        dev.to_csv(os.path.join(outdir, get_name('dev')))
        test.to_csv(os.path.join(outdir, get_name('test')))

    elif args.output_format == "parquet":
        train.to_parquet(os.path.join(outdir, get_name('train')))
        dev.to_parquet(os.path.join(outdir, get_name('dev')))
        test.to_parquet(os.path.join(outdir, get_name('test')))
