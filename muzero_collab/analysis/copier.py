import datetime
import os
import shutil
import time

import schedule

import muzero_collab.analysis.make_gif as mg
import muzero_collab.games.battle as battle


def copy(src_path, src_file='model.checkpoint', dest_path=None, dest_file=None, verbose=False):
    curr_date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')

    if not dest_path:
        dest_path = src_path

    if not dest_file:
        dest_file = f'model_{curr_date}.checkpoint'

    src_loc = os.path.join(src_path, src_file)
    target_loc = os.path.join(dest_path, dest_file)

    fn = shutil.copy(src_loc, target_loc)


    if verbose:
        print(f'Copy made at {curr_date} --- {fn}')

    return fn


def main(args):

    # function that will be ran repeatedly
    def f():
        filename = copy(args.experiment, src_file=args.filename, verbose=args.no_verbose)

        if args.gifs:
            gif_filename = filename.split('/')[-1].split('.')[0]
            mg.make_gif(filename, os.path.join(args.experiment, 'gifs'), map=args.map, filename=gif_filename, fps=25)


    # scheduling the function to occur at given frequency
    schedule.every(args.frequency).minutes.do(f)

    # performing the function immediately
    f()

    # infinite loop calling the function or printing time until next call
    while True:
        n_seconds = schedule.idle_seconds()

        if n_seconds > 0:
            fmt_time = str(datetime.timedelta(seconds=int(n_seconds)))
            print(f'Time til next copy: {fmt_time}', end='\r')
        else:
            schedule.run_pending()

        time.sleep(1)
 

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Makes periodic copies of checkpoint file throughout training')

    parser.add_argument('experiment', help='Path to experiment folder')
    parser.add_argument('--frequency', type=int, default=30, help='Frequency in minutes to save checkpoints (Default: 30 minutes)')
    parser.add_argument('--filename', default='model.checkpoint', help='Filename to make make copies of (Default: model.checkpoint')
    parser.add_argument('--gifs', action='store_true', help='Generate gifs for at each checkpoint (Default: False)')
    parser.add_argument('--map', default='empty', choices=battle.MAPS.keys(), help='Map to generate gif in (Default: empty)')
    parser.add_argument('--no-verbose', action='store_false', help='Turn off print statements')

    args = parser.parse_args()

    main(args)

