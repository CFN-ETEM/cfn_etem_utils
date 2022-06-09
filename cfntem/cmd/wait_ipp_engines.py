import argparse, os
import ipyparallel as ipp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--engines', type=int, required=True,
                        help='Number of engines')
    parser.add_argument('--ipp_dir', type=str, default='ipypar',
                        help='The directory for IpyParallel environment')
    args = parser.parse_args()

    c = ipp.Client(connection_info=f"{args.ipp_dir}/security/ipcontroller-client.json")

    c.wait_for_engines(n=args.engines, timeout=3600)
    
if __name__ == '__main__':
    main()
