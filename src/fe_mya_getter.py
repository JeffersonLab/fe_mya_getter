import argparse
import concurrent.futures
import logging
import os
import re
import subprocess
from datetime import datetime
from typing import List, Union, Dict

import numpy as np
import pandas as pd
import io

from tqdm import tqdm

app_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
app_name = "fe_mya_getter"
version = "1.0"

def get_data_log(file: str) -> pd.DataFrame:
    """Read in the data log file and return it as a DataFrame"""
    df = pd.read_csv(file, comment="#")

    # Convert timestamp columns to datetime
    df.settle_start = pd.to_datetime(df.settle_start)
    df.settle_end = pd.to_datetime(df.settle_end)
    df.avg_start = pd.to_datetime(df.avg_start)
    df.avg_end = pd.to_datetime(df.avg_end)

    # Rename some of the columns to have more context
    df.rename(mapper={'cavity_name': 'changed_cavity_name', 'cavity_epics_name': 'changed_epics_name'}, axis=1,
              inplace=True)

    return df


def get_mya_data_parallel(df: pd.DataFrame, pv_list: Union[List[str], None] = None,
                          pv_lags: Union[Dict[str, int], None] = None, include_settle: bool = False,
                          max_workers: int = 8) -> pd.DataFrame:
    """This uses a thread pool to call mySampler in subprocesses.


    """
    # Use a stock list of PVs and PV lags if none are provided
    if pv_list is None:
        # Add NDX readings
        pv_list = [f"INX{zone}_gDsRt" for zone in
                   ('1L05', '1L06', '1L07', '1L08', '1L22', '1L23', '1L24', '1L25', '1L26', '1L27')]
        pv_list += [f"INX{zone}_nDsRt" for zone in
                    ('1L05', '1L06', '1L07', '1L08', '1L22', '1L23', '1L24', '1L25', '1L26', '1L27')]
        pv_lags = {pv: -1 for pv in pv_list}

        # Add RF cavity info
        for lin in [1]:
            # Note that 1L26 (Q) was not present here
            for z in "2 3 4 5 6 7 8 9 A B C D E F G H I J K L M N O P".split():
                for c in range(1, 9):
                    pv_list += [f"R{lin}{z}{c}GSET", f"R{lin}{z}{c}GMES", f"R{lin}{z}{c}PSET", f"R{lin}{z}{c}ODVH"]

    # Call mySampler in parallel for each sample
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:

        # Submit a bunch of jobs to a pool.
        futures = []
        for i in range(len(df)):
            row = df.iloc[i, :]
            futures.append(
                executor.submit(row_to_mya_sample, row=row, pv_list=pv_list, pv_lags=pv_lags,
                                include_settle=include_settle))
        #  Wait for results.  tqdm gives a progress bar
        for future in tqdm(concurrent.futures.as_completed(futures)):
            results.append(future.result())

        # Concat all of the individual DFs back into one big one
        mya_df = pd.concat(results, ignore_index=True)

    return mya_df


def row_to_mya_sample(row: pd.DataFrame, pv_list: List[str], pv_lags: Dict[str, int],
                      include_settle: bool) -> pd.DataFrame:
    """Take a single row representing an fe_daq sample and query the MYA data for it.

    Args:
        row: A singled row DataFrame
        pv_list:  A list of PV names that are to be queried
        pv_lags:  A dictionary of PV names to number of samples (seconds) to lag those PVs by
        include_settle: Should settle time be included in the mya query or just the data
    """

    # Calculate the most we need to lag a PV by
    lags = [int(x) for x in pv_lags.values()]
    if np.max(lags) > 0:
        raise ValueError("Only non-positive lags (backwards looking) are supported.")
    max_lag = int(np.max(np.abs(lags)))

    # Get the parameters needed by mySampler
    n_samples = int(np.round(row.avg_dur))
    start = row.avg_start.ceil(freq='s').to_pydatetime().strftime("%Y-%m-%d %H:%M:%S")

    # We will label the samples based on which set they came from.  Plus 1 for boundaries.
    sample_type = ['average'] * (n_samples + 1)

    # They are a little different if we want to include the time that systems were settling
    if include_settle:
        start = row.settle_start.ceil(freq='s').to_pydatetime().strftime("%Y-%m-%d %H:%M:%S")
        n_settle = int(np.round(row.settle_dur))
        n_samples += n_settle
        sample_type = ['settle'] * n_settle + sample_type

    # Many signals (e.g., NDX dose rates) lag behind due to integrating measurements over the preceding second(s).
    # Add more sample so we can "lag" the samples later.  Add additional sample because of inclusive boundaries
    # Since we're already rounding up, maybe there is a problem?
    n_samples = n_samples + 1 + max_lag
    try:
        sample_df = get_single_sample(start, n_samples, pv_list)
    except subprocess.CalledProcessError as exc:
        print(f"Exception when calling mya on row with settle_start = {row.settle_start[0]}.\n{exc.output}")
        raise exc

    # Some of the PVs have a lag
    for pv in pv_lags.keys():
        sample_df[f"{pv}_lag{pv_lags[pv]}"] = sample_df[pv].shift(pv_lags[pv])

    # We only wanted the last <max_lag> rows to allow the appropriate shifting.  Drop the extra columns.
    sample_df.drop(sample_df.tail(max_lag).index, inplace=True)

    # Tag the type of sample
    sample_df.insert(1, 'sample_type', sample_type)

    # Add the start time of the sample for tracking purposes
    sample_df.insert(2, 'settle_start', row.settle_start)

    return sample_df


def get_single_sample(start: str, n_samples: int, pvlist: List[str]) -> pd.DataFrame:
    """Run mySampler with the specified arguments and return a single row DataFrame.

    The results for each PV is saved as a column with of tuples.  The tuples contain the individual samples in order.

    Raises:
        SubprocessError when something goes wrong with mySampler call
    """

    # Run the mySampler command to get samples at 1s intervals
    mysampler_cmd = '/usr/csite/certified/bin/mySampler'
    args = [mysampler_cmd, '-b', start, '-s', '1s', '-n', str(n_samples)] + pvlist
    logging.info(f"Starting {args[:7]} + {pvlist[0]}, ... ({len(pvlist)} PVs)")
    output = subprocess.run(args=args, check=True, capture_output=True)
    logging.info(f"Finished {args[:7]} + {pvlist[0]}, ... ({len(pvlist)} PVs)")
    lines = output.stdout.decode('UTF-8').split('\n')
    date_pattern = re.compile(r'^(\d\d\d\d-\d\d-\d\d) (\d.*)')
    space_pattern = re.compile(r'(\s+)')

    # mySampler returns a human readable format.  We want a CSV for pandas.
    processed_lines = []
    for line in lines:
        # Remove trailing or leading whitespace
        line = line.strip()

        # Modify the timestamp format
        line = re.sub(date_pattern, r"\1_\2", line)

        # Convert multiple spaces to a single comma to be CSV compatible
        line = re.sub(space_pattern, ',', line)

        processed_lines.append(line)

    # Create a single CSV string, then read it into a DataFrame
    csv_out = "\n".join(processed_lines)
    df = pd.read_csv(io.StringIO(csv_out))

    return df


def main():
    now = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    logging.basicConfig(level=logging.DEBUG, filename=f"{app_root}/log/fe_mya_getter-{now}.log", filemode="w",
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    logging.info("App starting")

    parser = argparse.ArgumentParser(description='Query archiver data collected during fe_daq use based on fe_daq data'
                                                 'index files, i.e. gradient-scan.csv')
    parser.add_argument('-v', '--version', action='version', version=f"{app_name} v{version}")
    parser.add_argument('-i', '--input-file', required=True, type=str,
                        help="Location of the input file to process.  I suggest keeping them in the data/ directory.")
    parser.add_argument('-o', '--output-file', type=str,
                        help="Path of output file.  Defaults to out/processed_<input_file_name>")
    parser.add_argument('-m', '--max-workers', type=int, default=8,
                        help="The maximum number of mySampler jobs to run in parallel.")

    # Process and validate the arguments
    args = parser.parse_args()
    file = args.input_file
    output_file = args.output_file
    max_workers = args.max_workers
    if output_file is None:
        output_file = f"{app_root}/out/processed_{os.path.basename(file)}"
    if not os.path.isfile(file):
        print(f"Error: Input file not found - {file}")
        exit(1)
    if not os.path.isdir(os.path.dirname(output_file)):
        print(f"Error: Path to output file not found - {output_file}")
        exit(1)
    if os.path.exists(output_file):
        print(f"Error: Output file already exists - {output_file}")
        exit(1)
    if max_workers < 1:
        print(f"Error: max_workers must be a positive integer")

    # Process the input file which contains information about the fe_daq samples
    df = get_data_log(file)

    # Run the parallel mya query
    mya_df = get_mya_data_parallel(df, include_settle=True, max_workers=max_workers)

    # Write the output file
    mya_df.to_csv(output_file, index=False)


if __name__ == "__main__":
    main()
